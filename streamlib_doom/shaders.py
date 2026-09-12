"""The three GLSL compute kernels of the Doom graph.

The renderer is one invocation per screen column, the way the 1993 engine
walked the screen: it casts a ray through every linedef, sorts the hits, and
walks them front to back drawing walls, flats and sky into a shrinking window,
then billboards the things against the depth it recorded. Everything it draws
is a palette index already passed through COLORMAP, so what it publishes is
the engine's own 8-bit framebuffer.
"""

RENDERER_GLSL = r"""#version 450
layout(local_size_x = 32, local_size_y = 1) in;
layout(set = 0, binding = 0) uniform sampler2D atlas;
layout(set = 0, binding = 1) uniform sampler2D level;
layout(set = 0, binding = 2) uniform sampler2D dynamic;
layout(set = 0, binding = 3) uniform sampler2D colormap;
layout(set = 0, binding = 4, rgba8) uniform writeonly image2D view_image;
layout(push_constant) uniform PC {
    float px; float py; float pz; float angle;
    float tick; float thing_count; float player_sector; float extralight;
    float linedef_count; float sky_entry; float unused0; float unused1;
} pc;

const int SCREEN_W = 320;
const int SCREEN_H = 200;
const float CENTER_Y = 100.0;
const float FOCAL = 160.0;
const int MAX_HITS = 48;
const float TWO_PI = 6.28318530718;

float depth_of_row[SCREEN_H];

int atlas_index(int entry, int tx, int ty) {
    vec4 rect = texelFetch(level, ivec2(2 * entry, 3), 0);
    return int(texelFetch(atlas, ivec2(int(rect.x) + tx, int(rect.y) + ty), 0).r * 255.0 + 0.5);
}
int atlas_alpha(int entry, int tx, int ty) {
    vec4 rect = texelFetch(level, ivec2(2 * entry, 3), 0);
    return int(texelFetch(atlas, ivec2(int(rect.x) + tx, int(rect.y) + ty), 0).g * 255.0 + 0.5);
}
int positive_mod(int a, int m) { int r = a % m; return r < 0 ? r + m : r; }

int light_map(float light, float dist) {
    int lightnum = clamp((int(light) >> 4) + int(pc.extralight), 0, 15);
    float startmap = float(15 - lightnum) * 4.0;
    float level_index = startmap - 1280.0 / max(dist, 1.0);
    return clamp(int(level_index), 0, 31);
}
int shade(int map_index, int palette_index) {
    return int(texelFetch(colormap, ivec2(palette_index, map_index), 0).r * 255.0 + 0.5);
}
// The view is rgba8: r is the palette index the compositor reads, g is depth on a log
// scale (4 map units at 0, 1024 at 255, sky at 255), b is the surface class
// (0 sky, 1 wall, 2 floor, 3 ceiling, 4 monster, 5 pickup, 6 decoration, 7 projectile).
float depth_code(float dist) { return clamp(log2(max(dist, 4.0) / 4.0) / 8.0, 0.0, 1.0); }
void put(int col, int y, int index, float dist, int kind) {
    imageStore(view_image, ivec2(col, y), vec4(float(index) / 255.0, depth_code(dist), float(kind) / 255.0, 1.0));
}

void draw_plane(int col, vec2 P, float ez, vec2 d, float row_top, float row_bottom, float plane_height,
                int flat_entry, bool is_sky, float light) {
    int y0 = max(0, int(ceil(row_top)));
    int y1 = min(SCREEN_H, int(ceil(row_bottom)));
    if (is_sky) {
        float ray_angle = atan(d.y, d.x);
        if (ray_angle < 0.0) ray_angle += TWO_PI;
        int sky_x = int(ray_angle / TWO_PI * 1024.0) & 255;
        for (int y = y0; y < y1; y++) {
            int sky_y = clamp(y, 0, 127);
            put(col, y, atlas_index(int(pc.sky_entry), sky_x, sky_y), 1.0e9, 0);
            depth_of_row[y] = 1.0e9;
        }
        return;
    }
    for (int y = y0; y < y1; y++) {
        float denom = CENTER_Y - (float(y) + 0.5);
        if (abs(denom) < 0.001) continue;
        float dist = (plane_height - ez) * FOCAL / denom;
        vec2 world = P + d * dist;
        int tx = int(floor(world.x)) & 63;
        int ty = int(floor(-world.y)) & 63;
        put(col, y, shade(light_map(light, dist), atlas_index(flat_entry, tx, ty)), dist, plane_height < ez ? 2 : 3);
        depth_of_row[y] = dist;
    }
}

void draw_wall(int col, float ez, float scale, float row_top, float row_bottom, int entry,
               float u_along, float tex_top_world, float light, float dist) {
    if (entry < 0) return;
    vec4 rect = texelFetch(level, ivec2(2 * entry, 3), 0);
    int w = int(rect.z), h = int(rect.w);
    int tx = positive_mod(int(floor(u_along)), w);
    int y0 = max(0, int(ceil(row_top)));
    int y1 = min(SCREEN_H, int(ceil(row_bottom)));
    int map_index = light_map(light, dist);
    for (int y = y0; y < y1; y++) {
        float world_h = ez + (CENTER_Y - (float(y) + 0.5)) / scale;
        int ty = positive_mod(int(floor(tex_top_world - world_h)), h);
        if (atlas_alpha(entry, tx, ty) == 0) continue;
        put(col, y, shade(map_index, atlas_index(entry, tx, ty)), dist, 1);
        depth_of_row[y] = dist;
    }
}

void main() {
    int col = int(gl_GlobalInvocationID.x);
    if (col >= SCREEN_W) return;
    vec2 P = vec2(pc.px, pc.py);
    float ez = pc.pz;
    vec2 f = vec2(cos(pc.angle), sin(pc.angle));
    vec2 r = vec2(f.y, -f.x);
    float s = (float(col) + 0.5 - 160.0) / FOCAL;
    vec2 d = f + r * s;

    for (int y = 0; y < SCREEN_H; y++) { depth_of_row[y] = 1.0e9; put(col, y, 0, 1.0e9, 0); }

    float hit_t[MAX_HITS]; float hit_u[MAX_HITS]; int hit_line[MAX_HITS];
    int hits = 0;
    int line_count = int(pc.linedef_count);
    for (int i = 0; i < line_count; i++) {
        vec4 a = texelFetch(level, ivec2(2 * i, 0), 0);
        vec2 v1 = a.xy, e = a.zw - a.xy;
        float denom = d.x * e.y - d.y * e.x;
        if (abs(denom) < 1.0e-7) continue;
        vec2 w = v1 - P;
        float t = (w.x * e.y - w.y * e.x) / denom;
        float u = (w.x * d.y - w.y * d.x) / denom;
        if (t <= 0.05 || u < 0.0 || u > 1.0) continue;
        if (hits < MAX_HITS) {
            int j = hits++;
            while (j > 0 && hit_t[j - 1] > t) { hit_t[j] = hit_t[j - 1]; hit_u[j] = hit_u[j - 1]; hit_line[j] = hit_line[j - 1]; j--; }
            hit_t[j] = t; hit_u[j] = u; hit_line[j] = i;
        } else if (t < hit_t[MAX_HITS - 1]) {
            int j = MAX_HITS - 1;
            while (j > 0 && hit_t[j - 1] > t) { hit_t[j] = hit_t[j - 1]; hit_u[j] = hit_u[j - 1]; hit_line[j] = hit_line[j - 1]; j--; }
            hit_t[j] = t; hit_u[j] = u; hit_line[j] = i;
        }
    }

    float ceil_clip = 0.0, floor_clip = float(SCREEN_H);
    for (int k = 0; k < hits; k++) {
        int i = hit_line[k];
        float t = hit_t[k];
        vec4 a = texelFetch(level, ivec2(2 * i, 0), 0);
        vec4 b = texelFetch(level, ivec2(2 * i + 1, 0), 0);
        vec2 v1 = a.xy, e = a.zw - a.xy;
        int flags = int(b.x), right_side = int(b.y), left_side = int(b.z);
        float side = e.x * (P.y - v1.y) - e.y * (P.x - v1.x);
        int front_side = (side < 0.0) ? right_side : left_side;
        int back_side = (side < 0.0) ? left_side : right_side;
        if (front_side < 0) { front_side = back_side; back_side = -1; }
        if (front_side < 0) continue;
        vec4 sd0 = texelFetch(level, ivec2(2 * front_side, 1), 0);
        vec4 sd1 = texelFetch(level, ivec2(2 * front_side + 1, 1), 0);
        int front = int(sd0.z);
        vec4 fh = texelFetch(dynamic, ivec2(front, 2), 0);     // floor, ceiling (doors and lifts move them)
        vec4 fdyn = texelFetch(dynamic, ivec2(front, 1), 0);   // light, floor flat, ceiling flat, sky flag
        float front_floor = fh.x, front_ceil = fh.y, light = fdyn.x;
        int floor_flat = int(fdyn.y), ceil_flat = int(fdyn.z);
        bool front_sky = fdyn.w > 0.5;
        float scale = FOCAL / t;
        float ceil_row = CENTER_Y - (front_ceil - ez) * scale;
        float floor_row = CENTER_Y - (front_floor - ez) * scale;

        draw_plane(col, P, ez, d, ceil_clip, min(ceil_row, floor_clip), front_ceil, ceil_flat, front_sky, light);
        draw_plane(col, P, ez, d, max(floor_row, ceil_clip), floor_clip, front_floor, floor_flat, false, light);

        // Special 48 is Doom's scrolling wall: the texture slides one texel per tic.
        float u_along = hit_u[k] * length(e) + sd0.x + ((int(b.w) == 48) ? pc.tick : 0.0);
        float yoff = sd0.y;
        int upper = int(sd1.x), lower = int(sd1.y), middle = int(sd1.z);
        bool two_sided = ((flags & 4) != 0) && back_side >= 0;
        if (!two_sided) {
            float tex_top = ((flags & 16) != 0) ? front_floor + texelFetch(level, ivec2(2 * max(middle, 0), 3), 0).w : front_ceil;
            draw_wall(col, ez, scale, max(ceil_row, ceil_clip), min(floor_row, floor_clip), middle, u_along, tex_top + yoff, light, t);
            break;
        }
        vec4 bsd0 = texelFetch(level, ivec2(2 * back_side, 1), 0);
        int back = int(bsd0.z);
        vec4 bh = texelFetch(dynamic, ivec2(back, 2), 0);
        vec4 bdyn = texelFetch(dynamic, ivec2(back, 1), 0);
        float back_floor = bh.x, back_ceil = bh.y;
        bool back_sky = bdyn.w > 0.5;
        float back_ceil_row = CENTER_Y - (back_ceil - ez) * scale;
        float back_floor_row = CENTER_Y - (back_floor - ez) * scale;
        bool sky_hack = front_sky && back_sky;
        if (back_ceil < front_ceil && !sky_hack && upper >= 0) {
            float tex_h = texelFetch(level, ivec2(2 * upper, 3), 0).w;
            float tex_top = ((flags & 8) != 0) ? front_ceil : back_ceil + tex_h;
            draw_wall(col, ez, scale, max(ceil_row, ceil_clip), min(back_ceil_row, floor_clip), upper, u_along, tex_top + yoff, light, t);
        }
        if (back_floor > front_floor && lower >= 0) {
            float tex_top = ((flags & 16) != 0) ? front_ceil : back_floor;
            draw_wall(col, ez, scale, max(back_floor_row, ceil_clip), min(floor_row, floor_clip), lower, u_along, tex_top + yoff, light, t);
        }
        if (middle >= 0) {
            float tex_h = texelFetch(level, ivec2(2 * middle, 3), 0).w;
            float open_top = max(front_ceil, back_ceil), open_bottom = max(front_floor, back_floor);
            float tex_top = ((flags & 16) != 0) ? open_bottom + tex_h : min(front_ceil, back_ceil);
            float top_row = CENTER_Y - (min(front_ceil, back_ceil) - ez) * scale;
            float bottom_row = CENTER_Y - (open_bottom - ez) * scale;
            draw_wall(col, ez, scale, max(top_row, ceil_clip), min(bottom_row, floor_clip), middle, u_along, tex_top + yoff, light, t);
        }
        ceil_clip = max(ceil_clip, sky_hack ? ceil_row : max(ceil_row, back_ceil_row));
        floor_clip = min(floor_clip, min(floor_row, back_floor_row));
        if (ceil_clip >= floor_clip) break;
    }

    // Things, already sorted far to near by the host.
    int things = int(pc.thing_count);
    for (int n = 0; n < things; n++) {
        vec4 t0 = texelFetch(dynamic, ivec2(2 * n, 0), 0);      // x, y, z, entry
        vec4 t1 = texelFetch(dynamic, ivec2(2 * n + 1, 0), 0);  // mirrored, light, fullbright, class
        vec2 rel = t0.xy - P;
        float depth = dot(rel, f);
        if (depth < 4.0) continue;
        float side_x = dot(rel, r);
        float scale = FOCAL / depth;
        int entry = int(t0.w);
        vec4 rect = texelFetch(level, ivec2(2 * entry, 3), 0);
        vec4 offs = texelFetch(level, ivec2(2 * entry + 1, 3), 0);
        int w = int(rect.z), h = int(rect.w);
        float x0 = 160.0 + (side_x - offs.x) * scale;
        float fx = (float(col) + 0.5 - x0) / scale;
        if (fx < 0.0 || fx >= float(w)) continue;
        int tx = int(fx);
        if (t1.x > 0.5) tx = w - 1 - tx;
        float top_world = t0.z + offs.y;
        float row_top = CENTER_Y - (top_world - ez) * scale;
        float row_bottom = row_top + float(h) * scale;
        int y0 = max(0, int(ceil(row_top))), y1 = min(SCREEN_H, int(ceil(row_bottom)));
        int map_index = (t1.z > 0.5) ? 0 : light_map(t1.y, depth);
        for (int y = y0; y < y1; y++) {
            if (depth >= depth_of_row[y]) continue;
            int ty = int((float(y) + 0.5 - row_top) / scale);
            if (ty < 0 || ty >= h) continue;
            if (atlas_alpha(entry, tx, ty) == 0) continue;
            put(col, y, shade(map_index, atlas_index(entry, tx, ty)), depth, int(t1.w));
        }
    }
}
"""

# One invocation per pixel of the 320x200 frame: the 3D view scaled into the
# 168-row viewport, the weapon lit like the room, then the status bar unlit.
COMPOSITOR_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D view_from_renderer;
layout(set = 0, binding = 1) uniform sampler2D atlas;
layout(set = 0, binding = 2) uniform sampler2D level;
layout(set = 0, binding = 3) uniform sampler2D hud;
layout(set = 0, binding = 4) uniform sampler2D colormap;
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D frame_image;
layout(push_constant) uniform PC {
    float draw_count; float weapon_map_index; float unused0; float unused1;
} pc;

const int VIEW_H = 168;
const int HUD_CLASS = 8;

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 320 || at.y >= 200) return;
    // Depth and class are read at the row the colour came from, so all three channels describe the
    // same pixel once the view is squashed into the viewport. A consumer that masks or reprojects on
    // them would otherwise be working against a picture shifted by 200/168.
    int index;
    float depth_code;
    int surface_class;
    if (at.y < VIEW_H) {
        int src_y = (at.y * 200) / VIEW_H;
        vec4 under = texelFetch(view_from_renderer, ivec2(at.x, src_y), 0);
        index = int(under.r * 255.0 + 0.5);
        depth_code = under.g;
        surface_class = int(under.b * 255.0 + 0.5);
    } else {
        index = 0;
        depth_code = 1.0;
        surface_class = HUD_CLASS;
    }
    int draws = int(pc.draw_count);
    for (int n = 0; n < draws; n++) {
        vec4 d0 = texelFetch(hud, ivec2(n, 0), 0);   // entry, x, y, lit (1 = shade like the room)
        vec4 rect = texelFetch(level, ivec2(2 * int(d0.x), 3), 0);
        int lx = at.x - int(d0.y), ly = at.y - int(d0.z);
        if (lx < 0 || ly < 0 || lx >= int(rect.z) || ly >= int(rect.w)) continue;
        ivec2 texel = ivec2(int(rect.x) + lx, int(rect.y) + ly);
        vec4 sample_ = texelFetch(atlas, texel, 0);
        if (sample_.g < 0.5) continue;
        int palette_index = int(sample_.r * 255.0 + 0.5);
        if (d0.w > 0.5) {
            palette_index = int(texelFetch(colormap, ivec2(palette_index, int(pc.weapon_map_index)), 0).r * 255.0 + 0.5);
        }
        index = palette_index;
        surface_class = HUD_CLASS;  // the weapon and the status bar are not part of the world
    }
    imageStore(frame_image, at, vec4(float(index) / 255.0, depth_code, float(surface_class) / 255.0, 1.0));
}
"""

# 320x200 indexed -> 1280x960 RGB through one of PLAYPAL's palettes, nearest
# neighbour with the 4:3 stretch a 1993 monitor applied.
UPSCALER_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D frame_from_compositor;
layout(set = 0, binding = 1) uniform sampler2D palettes;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_image;
layout(push_constant) uniform PC { float palette; float unused0; float unused1; float unused2; } pc;

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 1280 || at.y >= 960) return;
    ivec2 src = ivec2(at.x / 4, (at.y * 200) / 960);
    int index = int(texelFetch(frame_from_compositor, src, 0).r * 255.0 + 0.5);
    vec4 rgb = texelFetch(palettes, ivec2(index, int(pc.palette)), 0);
    imageStore(output_image, at, vec4(rgb.rgb, 1.0));
}
"""

# One workgroup that samples a texture and writes nothing anyone reads: binding
# the upscaler's own output as a sampled texture is what leaves it in the
# layout the H.264 encoder samples from.
SETTLE_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D settled_source;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D scratch_image;
void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    vec4 texel = texelFetch(settled_source, at, 0);
    imageStore(scratch_image, at, texel);
}
"""


# The game compositor: COMPOSITOR_GLSL plus a final effect stage — an index remap
# selected per frame, and crt scanlines/vignette — so one processor bakes the
# director's screen effect into the frame the phone and the recorder both see.
GAME_COMPOSITOR_GLSL = r"""#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D view_from_renderer;
layout(set = 0, binding = 1) uniform sampler2D atlas;
layout(set = 0, binding = 2) uniform sampler2D level;
layout(set = 0, binding = 3) uniform sampler2D hud;
layout(set = 0, binding = 4) uniform sampler2D colormap;
layout(set = 0, binding = 5) uniform sampler2D effect_remap;   // 256 x modes
layout(set = 0, binding = 6, rgba8) uniform writeonly image2D frame_image;
layout(push_constant) uniform PC { float draw_count; float weapon_map_index; float effect; float tick; } pc;

const int VIEW_H = 168;
const int HUD_CLASS = 8;
int shade(int map_index, int palette_index) { return int(texelFetch(colormap, ivec2(palette_index, map_index), 0).r * 255.0 + 0.5); }

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    if (at.x >= 320 || at.y >= 200) return;
    // Depth and class are read at the row the colour came from, so all three channels describe the
    // same pixel once the view is squashed into the viewport. A consumer that masks or reprojects on
    // them would otherwise be working against a picture shifted by 200/168.
    int index;
    float depth_code;
    int surface_class;
    if (at.y < VIEW_H) {
        int src_y = (at.y * 200) / VIEW_H;
        vec4 under = texelFetch(view_from_renderer, ivec2(at.x, src_y), 0);
        index = int(under.r * 255.0 + 0.5);
        depth_code = under.g;
        surface_class = int(under.b * 255.0 + 0.5);
    } else {
        index = 0;
        depth_code = 1.0;
        surface_class = HUD_CLASS;
    }
    int draws = int(pc.draw_count);
    for (int n = 0; n < draws; n++) {
        vec4 d0 = texelFetch(hud, ivec2(n, 0), 0);
        vec4 rect = texelFetch(level, ivec2(2 * int(d0.x), 3), 0);
        int lx = at.x - int(d0.y), ly = at.y - int(d0.z);
        if (lx < 0 || ly < 0 || lx >= int(rect.z) || ly >= int(rect.w)) continue;
        vec4 sample_ = texelFetch(atlas, ivec2(int(rect.x) + lx, int(rect.y) + ly), 0);
        if (sample_.g < 0.5) continue;
        int palette_index = int(sample_.r * 255.0 + 0.5);
        if (d0.w > 0.5) palette_index = int(texelFetch(colormap, ivec2(palette_index, int(pc.weapon_map_index)), 0).r * 255.0 + 0.5);
        index = palette_index;
        surface_class = HUD_CLASS;  // the weapon and the status bar are not part of the world
    }
    int mode = int(pc.effect);
    if (mode != 0) {
        index = int(texelFetch(effect_remap, ivec2(index, mode), 0).r * 255.0 + 0.5);
        if (mode == 1 && at.y < VIEW_H) {                                  // crt over the view only
            float dx = (float(at.x) - 160.0) / 160.0, dy = (float(at.y) - 84.0) / 84.0;
            int darken = int(clamp((dx * dx + dy * dy) * 10.0, 0.0, 12.0));
            if ((at.y & 1) == 1) darken += 5;
            index = shade(darken, index);
        }
    }
    imageStore(frame_image, at, vec4(float(index) / 255.0, depth_code, float(surface_class) / 255.0, 1.0));
}
"""
