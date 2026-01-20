"""Streamlit app: Bernoulli's Principle + Mass Continuity demonstration.

This digital model is an original construction that:
- Explains and visualizes Fluids in Motion & Bernoulli's Principle
- Explains and visualizes Mass Flow / Mass Continuity
- Provides interactive sliders for physical parameters
- Produces quantitative numerical outputs and downloadable data

External libraries used:
- streamlit (web UI)
- numpy (numeric helpers)
- pandas (tabular data and CSV export)
- matplotlib (simple schematic visualization)

If you reuse this code, please cite it appropriately and also
cite the documentation of Streamlit, NumPy, Pandas, and Matplotlib.
"""

import math  # for pi and basic math operations
import time  # for simple animation timing
from typing import Tuple

import matplotlib.pyplot as plt  # for the schematic visualization
import numpy as np  # for convenient numeric helpers
import pandas as pd  # for tabular data and CSV export
import streamlit as st  # Streamlit for the interactive web app
import streamlit.components.v1 as components  # Embed custom 3D HTML/JS


# ----------------------------
# Core physics helper functions
# ----------------------------

def circle_area_from_diameter(d_m: float) -> float:
    """Return cross-sectional area (m^2) of a circular pipe from diameter in meters."""
    # Radius is half the diameter
    r = d_m / 2.0
    # Area of a circle: A = π r^2
    return math.pi * r * r


def compute_flow_and_pressures(
    rho: float,
    Q_m3s: float,
    d1_m: float,
    d2_m: float,
    p1_pa: float,
    dh_m: float,
    g: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Compute areas, velocities, mass flow, pressures, and continuity residual.

    This function implements:
    - Continuity equation: A1 * v1 = A2 * v2 = Q
    - Bernoulli equation between 1 and 2:
      p1 + 0.5*rho*v1^2 + rho*g*h1 = p2 + 0.5*rho*v2^2 + rho*g*h2

    Parameters
    ----------
    rho : float
        Fluid density in kg/m^3.
    Q_m3s : float
        Volumetric flow rate in m^3/s.
    d1_m : float
        Upstream pipe diameter in meters.
    d2_m : float
        Constriction pipe diameter in meters.
    p1_pa : float
        Upstream static pressure in Pascals (Pa).
    dh_m : float
        Height difference h2 - h1 in meters. Positive means section 2 is higher.
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    A1, A2, v1, v2, m_dot, p2, dp, continuity_residual, Q_m3s
    """
    # Compute cross-sectional areas from diameters
    A1 = circle_area_from_diameter(d1_m)
    A2 = circle_area_from_diameter(d2_m)

    # Continuity: use volumetric flow Q to compute velocities
    # Q = A * v  =>  v = Q / A
    v1 = Q_m3s / A1
    v2 = Q_m3s / A2

    # Mass flow rate: m_dot = rho * Q
    m_dot = rho * Q_m3s

    # Continuity residual: difference between A1*v1 and A2*v2
    # In ideal steady incompressible flow, this should be zero
    continuity_residual = abs(A1 * v1 - A2 * v2)

    # Bernoulli equation to solve for p2
    # p1 + 0.5*rho*v1^2 + rho*g*h1 = p2 + 0.5*rho*v2^2 + rho*g*h2
    # Let h2 = h1 + dh => h2 - h1 = dh
    # Rearranged for p2:
    # p2 = p1 + 0.5*rho*(v1^2 - v2^2) - rho*g*dh
    p2 = p1_pa + 0.5 * rho * (v1 ** 2 - v2 ** 2) - rho * g * dh_m
    dp = p2 - p1_pa

    return A1, A2, v1, v2, m_dot, p2, dp, continuity_residual, Q_m3s


def draw_schematic(
    d1_m: float,
    d2_m: float,
    v1: float,
    v2: float,
    p1_pa: float,
    p2_pa: float,
    particle_positions=None,
) -> plt.Figure:
    """Create a Matplotlib figure showing a simple pipe schematic.

    - Horizontal pipe with a wider upstream section and a narrower constriction
    - Arrow lengths proportional to velocity magnitude
    - Bar-style indicators for pressure at section 1 and section 2
    """
    # Create a figure and axis with a wide aspect ratio
    fig, ax = plt.subplots(figsize=(8, 3))

    # Clear axis ticks so the picture looks less like raw data
    ax.set_xticks([])
    ax.set_yticks([])

    # Define geometric positions for the drawing
    x_start = 0.05
    x_mid = 0.5
    x_end = 0.95
    y_center = 0.5

    # Map diameters (meters) to a relative visual height (0..1 range)
    # Use a simple scale factor so they fit nicely in the figure.
    scale = 3.0
    h1 = d1_m * scale
    h2 = d2_m * scale

    # Draw upstream pipe section as a rectangle
    ax.add_patch(
        plt.Rectangle(
            (x_start, y_center - h1 / 2.0),
            width=(x_mid - x_start),
            height=h1,
            color="#1f2937",
        )
    )

    # Draw downstream (constricted) pipe section as another rectangle
    ax.add_patch(
        plt.Rectangle(
            (x_mid, y_center - h2 / 2.0),
            width=(x_end - x_mid),
            height=h2,
            color="#111827",
        )
    )

    # Draw a simple tapered connector between the two sections
    ax.add_patch(
        plt.Polygon(
            [
                (x_mid, y_center - h1 / 2.0),
                (x_mid, y_center - h2 / 2.0),
                (x_mid, y_center + h2 / 2.0),
                (x_mid, y_center + h1 / 2.0),
            ],
            color="#1f2937",
        )
    )

    # Helper to draw arrows for velocity, with length proportional to speed
    def draw_velocity_arrow(x_center: float, v: float, color: str):
        """Draw a horizontal arrow at y_center with length scaled by velocity."""
        # Base length and scaling for visualization
        base_len = 0.1
        # Limit scale to avoid extremely long arrows
        arrow_scale = min(4.0, max(0.2, abs(v) / 2.0))
        half_len = base_len * arrow_scale

        # Arrow from left to right
        ax.arrow(
            x_center - half_len,
            y_center,
            2 * half_len,
            0.0,
            head_width=0.03,
            head_length=0.02,
            length_includes_head=True,
            color=color,
        )

    # Draw velocity arrows in each section (Bernoulli + Continuity visualization)
    draw_velocity_arrow((x_start + x_mid) / 2.0, v1, color="#22c55e")
    draw_velocity_arrow((x_mid + x_end) / 2.0, v2, color="#3b82f6")

    # Helper to draw simple vertical pressure bars
    def draw_pressure_bar(x_center: float, pressure_pa: float, label: str):
        """Draw a vertical bar representing static pressure."""
        # Convert pressure in Pa to kPa for scaling
        pressure_kpa = pressure_pa / 1000.0
        # Map pressure to bar height
        bar_height = min(0.4, max(0.05, pressure_kpa / 1000.0))
        ax.add_patch(
            plt.Rectangle(
                (x_center - 0.015, 0.05),
                width=0.03,
                height=bar_height,
                color="#ef4444",
            )
        )
        ax.text(
            x_center,
            0.02,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    # Draw upstream and downstream pressure indicators
    draw_pressure_bar(x_start + 0.05, p1_pa, "p₁")
    draw_pressure_bar(x_end - 0.05, p2_pa, "p₂")

    # Add simple explanatory labels
    ax.text(
        (x_start + x_mid) / 2.0,
        y_center + 0.4,
        "Section 1: larger area → lower speed\n(higher static pressure)",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
    )
    ax.text(
        (x_mid + x_end) / 2.0,
        y_center + 0.4,
        "Section 2: smaller area → higher speed\n(lower static pressure)",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
    )

    # Optionally draw tracer particles to create a live-flow feel in 2D
    if particle_positions is not None:
        # Place particles along the pipe centerline at y_center
        y_center = 0.5
        particle_positions = np.array(particle_positions, dtype=float)
        ys = np.full_like(particle_positions, y_center)
        ax.scatter(
            particle_positions,
            ys,
            s=18,
            color="#22c55e",
            alpha=0.9,
            edgecolors="none",
        )

    # Set limits of the drawing space and background color
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    return fig


def draw_pipe_3d_frame(
    d1_m: float,
    d2_m: float,
    v1: float,
    v2: float,
    p1_pa: float,
    p2_pa: float,
    particle_positions: np.ndarray,
) -> plt.Figure:
    """Create a 3D figure of a horizontal pipe with flowing tracer particles.

    - The pipe starts wide, narrows in the middle, then widens again.
    - Pipe radius and pressure color both change smoothly along the length.
    - Tracer particles move along the centerline from left to right.
    """
    # Create a 3D figure
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111, projection="3d")

    # Define basic geometry along the x-axis (0 to 1 is full pipe length)
    n_x = 40
    n_theta = 24
    x_vals = np.linspace(0.0, 1.0, n_x)
    theta_vals = np.linspace(0.0, 2.0 * math.pi, n_theta)

    # Radii: wide on both ends, narrow in the middle
    r1 = d1_m / 2.0
    r2 = d2_m / 2.0

    # Smooth radius profile: r(x) is smallest at x=0.5, largest at ends
    def radius_at_x(x: float) -> float:
        t = abs(x - 0.5) * 2.0
        if t > 1.0:
            t = 1.0
        return r2 + (r1 - r2) * t

    radii = np.array([radius_at_x(x) for x in x_vals])

    # Build surface coordinates for the pipe wall
    X, Theta = np.meshgrid(x_vals, theta_vals)
    R = np.tile(radii, (n_theta, 1))
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)

    # Approximate pressure variation along the pipe for coloring
    # Lower pressure near the middle where radius is smallest
    def pressure_at_x(x: float) -> float:
        t = abs(x - 0.5) * 2.0
        if t > 1.0:
            t = 1.0
        return p2_pa + (p1_pa - p2_pa) * t

    pressures = np.array([pressure_at_x(x) for x in x_vals])
    P = np.tile(pressures, (n_theta, 1))

    # Normalize pressures for colormap
    p_min = P.min()
    p_max = P.max()
    denom = p_max - p_min if p_max > p_min else 1.0
    P_norm = (P - p_min) / denom

    # Map pressures to colors (high pressure = warm, low pressure = cool)
    colors = plt.cm.coolwarm(P_norm)

    # Plot the pipe surface
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=True,
    )

    # Plot tracer particles along the centerline inside the pipe
    # Positions are given as x in [0, 1]; place them at y=z=0 for clarity.
    particle_positions = np.array(particle_positions, dtype=float)
    xp = particle_positions
    yp = np.zeros_like(xp)
    zp = np.zeros_like(xp)
    ax.scatter(
        xp,
        yp,
        zp,
        color="#22c55e",
        s=20,
        depthshade=False,
    )

    # Adjust view and aesthetics
    ax.set_xlim(0.0, 1.0)
    # Use symmetric limits in y and z based on maximum radius
    r_max = max(r1, r2)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 0.5, 0.5))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    return fig


def build_pipe_2d_html(
    *,
    rho: float,
    Q_L_s: float,
    d_left_cm: float,
    d_mid_cm: float,
    d_right_cm: float,
    p1_kPa: float,
    g: float,
    h_left_m: float,
    h_right_m: float,
    paused: bool,
    show_dots: bool,
    show_streaks: bool,
) -> str:
    """Return a self-contained HTML canvas animation that matches the 2D style reference.

    - No external scripts (avoids webview ORB restrictions).
    - Animated water is shown as moving streaks (not dots).
    - Pipe is semi-transparent so the blue water is visible inside.
    """
    # Cache bust to avoid stale webview content when the app reruns
    cache_bust = int(time.time())

    # Convert numeric values to JS-safe literals
    paused_js = "true" if paused else "false"

    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
      }}
      #wrap {{
        width: 100%;
        height: 520px;
        position: relative;
      }}
      canvas {{
        width: 100%;
        height: 520px;
        display: block;
        border-radius: 10px;
        background: #a9ddf4;
      }}
    </style>
  </head>
  <body>
    <div id="wrap">
      <canvas id="c"></canvas>
    </div>
    <script>
      // Cache bust: {cache_bust}
      const PARAMS = {{
        rho: {rho:.6f},        // kg/m^3
        Q_L_s: {Q_L_s:.6f},    // L/s
        dL_cm: {d_left_cm:.6f},
        dM_cm: {d_mid_cm:.6f},
        dR_cm: {d_right_cm:.6f},
        p1_kPa: {p1_kPa:.6f},
        g: {g:.6f},            // m/s^2
        hL_m: {h_left_m:.6f},  // m
        hR_m: {h_right_m:.6f}, // m
        paused: {paused_js},
        showDots: {"true" if show_dots else "false"},
        showStreaks: {"true" if show_streaks else "false"},
      }};

      const canvas = document.getElementById('c');
      const ctx = canvas.getContext('2d');

      function resize() {{
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }}
      window.addEventListener('resize', resize);
      resize();

      function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
      function lerp(a, b, t) {{ return a + (b - a) * t; }}

      function smoothstep(t) {{
        t = clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
      }}

      // Pipe radius profile: left → middle → right, smooth taper
      function radiusAtX01(x01) {{
        const rL = (PARAMS.dL_cm / 100.0) / 2.0;
        const rM = (PARAMS.dM_cm / 100.0) / 2.0;
        const rR = (PARAMS.dR_cm / 100.0) / 2.0;
        if (x01 <= 0.5) {{
          const t = smoothstep(x01 / 0.5);
          return lerp(rL, rM, t);
        }}
        const t = smoothstep((x01 - 0.5) / 0.5);
        return lerp(rM, rR, t);
      }}

      // Pipe centerline height profile (left end and right end can move up/down)
      function hAtX01_m(x01) {{
        return lerp(PARAMS.hL_m, PARAMS.hR_m, clamp(x01, 0.0, 1.0));
      }}

      // Continuity: v(x) = Q / A(x)
      function velAtX01(x01) {{
        const Q = PARAMS.Q_L_s / 1000.0; // m^3/s
        const r = radiusAtX01(x01);
        const A = Math.PI * r * r;
        return Q / Math.max(A, 1e-9);
      }}

      // Bernoulli pressure estimate relative to upstream (ideal, no losses).
      // Uses p + 1/2 ρ v^2 + ρ g h = constant, anchored at x01 ≈ 0.15.
      function pressureAtX01_kPa(x01) {{
        const v1 = velAtX01(0.15);
        const v = velAtX01(x01);
        const h1 = hAtX01_m(0.15);
        const h = hAtX01_m(x01);
        const p = (PARAMS.p1_kPa * 1000.0)
          + 0.5 * PARAMS.rho * (v1*v1 - v*v)
          + PARAMS.rho * PARAMS.g * (h1 - h);
        return p / 1000.0;
      }}

      // Utility for soft UI callouts (tooltip).
      function drawRoundedRect(x, y, w, h, r, fillStyle, strokeStyle) {{
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
        if (fillStyle) {{
          ctx.fillStyle = fillStyle;
          ctx.fill();
        }}
        if (strokeStyle) {{
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = 2;
          ctx.stroke();
        }}
      }}

      let last = performance.now();
      let phase = 0;
      const dots = [];
      const nDots = 260;
      for (let i = 0; i < nDots; i++) {{
        dots.push({{
          x01: 0.02 + Math.random() * 0.96,
          yN: lerp(-1, 1, Math.random()),
          seed: Math.random() * 1000
        }});
      }}

      // Pointer location used for the in-pipe probe tooltip.
      // We use pointer events so this works with both mouse and touch.
      let pointer = null;
      function setPointerFromEvent(e) {{
        const rect = canvas.getBoundingClientRect();
        pointer = {{
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        }};
      }}
      canvas.addEventListener('pointermove', (e) => {{
        setPointerFromEvent(e);
      }});
      canvas.addEventListener('pointerdown', (e) => {{
        setPointerFromEvent(e);
      }});
      canvas.addEventListener('pointerleave', () => {{
        pointer = null;
      }});

      function render(now) {{
        const dt = Math.min((now - last) / 1000.0, 0.05);
        last = now;
        if (!PARAMS.paused) {{
          // Move the water streaks; scale phase by flow rate
          phase += dt * (0.8 + PARAMS.Q_L_s / 20.0);
        }}

        const W = canvas.getBoundingClientRect().width;
        const H = canvas.getBoundingClientRect().height;

        // Background
        ctx.clearRect(0, 0, W, H);
        const bg = ctx.createLinearGradient(0, 0, 0, H);
        bg.addColorStop(0.0, '#0b1220');
        bg.addColorStop(1.0, '#070b14');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, W, H);
        const glow = ctx.createRadialGradient(W * 0.25, H * 0.15, 10, W * 0.25, H * 0.15, H * 0.9);
        glow.addColorStop(0.0, 'rgba(59,130,246,0.16)');
        glow.addColorStop(1.0, 'rgba(59,130,246,0.00)');
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, W, H);

        // Pipe placement
        const pipeX = W * 0.05;
        const pipeY = H * 0.56;
        const pipeW = W * 0.90;
        const hScale = H * 0.06; // px per meter for elevation

        // Convert radius (m) to pixels
        const maxR = Math.max(
          (PARAMS.dL_cm / 100.0) / 2.0,
          (PARAMS.dM_cm / 100.0) / 2.0,
          (PARAMS.dR_cm / 100.0) / 2.0
        );
        const pxPerM = (H * 0.22) / Math.max(maxR, 1e-6);
        const wall = 14; // pipe wall thickness in px

        // Build pipe inner shape path
        function buildPipePath(isOuter) {{
          const n = 120;
          ctx.beginPath();
          for (let i = 0; i <= n; i++) {{
            const x01 = i / n;
            const x = pipeX + pipeW * x01;
            const r = radiusAtX01(x01) * pxPerM + (isOuter ? wall : 0);
            const yCenter = pipeY - hAtX01_m(x01) * hScale;
            const yTop = yCenter - r;
            if (i === 0) ctx.moveTo(x, yTop);
            else ctx.lineTo(x, yTop);
          }}
          for (let i = n; i >= 0; i--) {{
            const x01 = i / n;
            const x = pipeX + pipeW * x01;
            const r = radiusAtX01(x01) * pxPerM + (isOuter ? wall : 0);
            const yCenter = pipeY - hAtX01_m(x01) * hScale;
            const yBot = yCenter + r;
            ctx.lineTo(x, yBot);
          }}
          ctx.closePath();
        }}

        function strokePipeEdge(isOuter, side, strokeStyle, width) {{
          const n = 140;
          ctx.beginPath();
          for (let i = 0; i <= n; i++) {{
            const x01 = i / n;
            const x = pipeX + pipeW * x01;
            const r = radiusAtX01(x01) * pxPerM + (isOuter ? wall : 0);
            const yCenter = pipeY - hAtX01_m(x01) * hScale;
            const y = side === 'top' ? (yCenter - r) : (yCenter + r);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }}
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = width;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.stroke();
        }}

        // Soft shadow under the pipe to make it feel grounded
        ctx.save();
        buildPipePath(true);
        ctx.shadowColor = 'rgba(0,0,0,0.55)';
        ctx.shadowBlur = 22;
        ctx.shadowOffsetY = 14;
        ctx.fillStyle = 'rgba(0,0,0,0.01)';
        ctx.fill();
        ctx.restore();

        // Outer pipe body (cylindrical shading)
        buildPipePath(true);
        const pipeGrad = ctx.createLinearGradient(0, pipeY - 140, 0, pipeY + 140);
        pipeGrad.addColorStop(0.00, 'rgba(245, 250, 255, 0.95)');
        pipeGrad.addColorStop(0.18, 'rgba(190, 210, 230, 0.92)');
        pipeGrad.addColorStop(0.50, 'rgba(95, 120, 150, 0.92)');
        pipeGrad.addColorStop(0.78, 'rgba(55, 70, 90, 0.92)');
        pipeGrad.addColorStop(1.00, 'rgba(30, 40, 55, 0.92)');
        ctx.fillStyle = pipeGrad;
        ctx.fill();

        // Subtle longitudinal reflection (gives a round, metallic feel)
        ctx.save();
        buildPipePath(true);
        ctx.clip();
        const reflect = ctx.createLinearGradient(pipeX, 0, pipeX + pipeW, 0);
        reflect.addColorStop(0.00, 'rgba(255,255,255,0.00)');
        reflect.addColorStop(0.25, 'rgba(255,255,255,0.05)');
        reflect.addColorStop(0.50, 'rgba(255,255,255,0.00)');
        reflect.addColorStop(0.78, 'rgba(59,130,246,0.05)');
        reflect.addColorStop(1.00, 'rgba(255,255,255,0.00)');
        ctx.fillStyle = reflect;
        ctx.fillRect(pipeX, 0, pipeW, H);
        ctx.restore();

        // Outer outline and edge shading
        ctx.strokeStyle = 'rgba(2, 6, 23, 0.75)';
        ctx.lineWidth = 4;
        ctx.stroke();
        strokePipeEdge(true, 'top', 'rgba(255,255,255,0.22)', 3);
        strokePipeEdge(true, 'bottom', 'rgba(0,0,0,0.28)', 3);

        // Inner rim strokes to show wall thickness
        strokePipeEdge(false, 'top', 'rgba(255,255,255,0.14)', 2);
        strokePipeEdge(false, 'bottom', 'rgba(0,0,0,0.18)', 2);

        // End caps (flanges) drawn as circles because we're viewing a cutaway side view.
        // These are decorative geometry cues to make the tube read as cylindrical.
        function drawEndCap(x01) {{
          const x = pipeX + pipeW * x01;
          const yCenter = pipeY - hAtX01_m(x01) * hScale;
          const rInner = radiusAtX01(x01) * pxPerM;
          const rOuter = rInner + wall;
          const rFlange = rOuter + 8;

          // Flange ring
          const rg = ctx.createRadialGradient(
            x - rFlange * 0.25,
            yCenter - rFlange * 0.35,
            rFlange * 0.1,
            x,
            yCenter,
            rFlange
          );
          rg.addColorStop(0.0, 'rgba(245, 250, 255, 0.95)');
          rg.addColorStop(0.5, 'rgba(115, 140, 170, 0.92)');
          rg.addColorStop(1.0, 'rgba(30, 40, 55, 0.92)');
          ctx.beginPath();
          ctx.arc(x, yCenter, rFlange, 0, Math.PI * 2);
          ctx.fillStyle = rg;
          ctx.fill();
          ctx.strokeStyle = 'rgba(2, 6, 23, 0.8)';
          ctx.lineWidth = 3;
          ctx.stroke();

          // Inner lip
          ctx.beginPath();
          ctx.arc(x, yCenter, rOuter, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255,255,255,0.18)';
          ctx.lineWidth = 2;
          ctx.stroke();

          // Hollow opening
          const hole = ctx.createRadialGradient(x - rInner * 0.15, yCenter - rInner * 0.15, rInner * 0.1, x, yCenter, rInner);
          hole.addColorStop(0.0, 'rgba(2, 6, 23, 0.75)');
          hole.addColorStop(1.0, 'rgba(2, 6, 23, 0.25)');
          ctx.beginPath();
          ctx.arc(x, yCenter, rInner, 0, Math.PI * 2);
          ctx.fillStyle = hole;
          ctx.fill();

          // A small highlight on the flange
          ctx.beginPath();
          ctx.arc(x - rFlange * 0.22, yCenter - rFlange * 0.22, rFlange * 0.14, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255,255,255,0.10)';
          ctx.fill();
        }}
        drawEndCap(0.0);
        drawEndCap(1.0);

        // Cut overlap so end caps don't render "through" the pipe body silhouette.
        // destination-out erases any part of the end caps that lies inside the pipe outline.
        ctx.save();
        buildPipePath(true);
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'rgba(0,0,0,1)';
        ctx.fill();
        ctx.restore();

        // Inner cavity clip
        buildPipePath(false);
        ctx.save();
        ctx.clip();

        // Water fill (depth gradient)
        const waterGrad = ctx.createLinearGradient(0, pipeY - 130, 0, pipeY + 130);
        waterGrad.addColorStop(0.0, 'rgba(110, 210, 255, 0.70)');
        waterGrad.addColorStop(0.5, 'rgba(60, 170, 255, 0.72)');
        waterGrad.addColorStop(1.0, 'rgba(20, 120, 255, 0.72)');
        ctx.fillStyle = waterGrad;
        ctx.fillRect(pipeX, pipeY - 240, pipeW, 480);

        // Cylindrical interior lighting: brighter centerline, darker near walls
        ctx.save();
        const centerGlow = ctx.createLinearGradient(0, pipeY - 120, 0, pipeY + 120);
        centerGlow.addColorStop(0.0, 'rgba(255,255,255,0.00)');
        centerGlow.addColorStop(0.45, 'rgba(255,255,255,0.10)');
        centerGlow.addColorStop(0.55, 'rgba(255,255,255,0.10)');
        centerGlow.addColorStop(1.0, 'rgba(255,255,255,0.00)');
        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = centerGlow;
        ctx.fillRect(pipeX, 0, pipeW, H);
        ctx.restore();

        // Darken near the inner walls to sell depth
        strokePipeEdge(false, 'top', 'rgba(2,6,23,0.18)', 6);
        strokePipeEdge(false, 'bottom', 'rgba(2,6,23,0.22)', 7);

        // Subtle inner specular highlight near the top wall
        strokePipeEdge(false, 'top', 'rgba(255,255,255,0.10)', 3);

        // Moving streaks (optional)
        if (PARAMS.showStreaks) {{
          const nStreaks = 18;
          for (let s = 0; s < nStreaks; s++) {{
            const yN = lerp(-0.75, 0.75, (s + 0.5) / nStreaks);
            ctx.beginPath();
            const amp = 4 + (s % 3) * 2;
            const freq = 9 + (s % 5) * 2;
            for (let i = 0; i <= 120; i++) {{
              const x01 = i / 120;
              const x = pipeX + pipeW * x01;
              const yCenter = pipeY - hAtX01_m(x01) * hScale;
              const rPx = radiusAtX01(x01) * pxPerM * 0.65;
              const baseY = yCenter + yN * rPx;
              const v = velAtX01(x01);
              const vRef = Math.max(velAtX01(0.15), 1e-6);
              const localPhase = phase * (v / vRef);
              const y = baseY + Math.sin((x01 * freq) + localPhase + s) * amp;
              if (i === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }}
            ctx.strokeStyle = 'rgba(255,255,255,0.18)';
            ctx.lineWidth = 2;
            ctx.stroke();
          }}
        }}

        // Moving water dots (optional). We approximate a laminar profile by moving dots
        // faster near the centerline and slower near the walls.
        if (PARAMS.showDots) {{
          const vRef = Math.max(velAtX01(0.15), 1e-6);
          const base = 0.08 + PARAMS.Q_L_s / 60.0;
          const warpAmp = 0.05;
          const twoPi = Math.PI * 2.0;

          if (!PARAMS.paused) {{
            for (let i = 0; i < dots.length; i++) {{
              const d = dots[i];
              const v = velAtX01(d.x01);
              const radial = clamp(Math.abs(d.yN), 0.0, 1.0);
              // Parabolic-ish profile: max at center (radial=0), approaches 0 at walls.
              const profile = Math.max(0.12, 1.0 - radial * radial);
              d.x01 += dt * base * (v / vRef) * (0.35 + 0.65 * profile);
              if (d.x01 > 1.02) d.x01 = -0.02;
              const jitter = PARAMS.Q_L_s > 0 ? 0.01 : 0.0;
              d.yN += (Math.sin(d.seed + phase * 0.7) * jitter) * dt;
              d.yN = clamp(d.yN, -1.0, 1.0);
            }}
          }}

          for (let i = 0; i < dots.length; i++) {{
            const d = dots[i];
            let x01 = d.x01;
            // Rendering warp (adds a mild bunching effect toward the constriction)
            x01 = clamp(x01 + warpAmp * Math.sin(2.0 * Math.PI * x01), 0.0, 1.0);
            const x = pipeX + pipeW * x01;
            const yCenter = pipeY - hAtX01_m(x01) * hScale;
            const rPx = radiusAtX01(x01) * pxPerM * 0.82;
            // Small streamline curvature for visual richness (kept subtle to preserve the
            // impression of steady flow along a tube).
            const bend = 0.22 * (1.0 - d.yN * d.yN) * Math.sin(twoPi * (x01 * 0.85) - phase * 0.35 + d.seed);
            const yN = clamp(d.yN + bend * 0.25, -1.0, 1.0);
            const y = yCenter + yN * rPx * 0.78;

            const radial = clamp(Math.abs(yN), 0.0, 1.0);
            const profile = Math.max(0.12, 1.0 - radial * radial);
            // Use localSpeed to drive particle appearance (size/alpha/blur) so faster flow
            // reads more energetic in the constricted section.
            const localSpeed = (velAtX01(x01) / vRef) * (0.35 + 0.65 * profile);
            const size = 1.2 + 2.4 * localSpeed;
            const alpha = 0.22 + 0.40 * localSpeed;

            ctx.save();
            ctx.shadowColor = 'rgba(40,150,255,0.35)';
            ctx.shadowBlur = 2.0 + 6.0 * localSpeed;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(110, 210, 255, ${{alpha.toFixed(3)}})`;
            ctx.fill();

            // Tiny bright core for fast particles
            if (localSpeed > 0.9) {{
              ctx.shadowBlur = 0;
              ctx.beginPath();
              ctx.arc(x, y, Math.max(0.9, size * 0.35), 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(255,255,255,0.28)';
              ctx.fill();
            }}
            ctx.restore();
          }}
        }}

        ctx.restore();

        // Inner border for crisp pipe edge
        buildPipePath(false);
        ctx.strokeStyle = 'rgba(0,0,0,0.35)';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Vignette to focus attention and emphasize depth
        const vignette = ctx.createRadialGradient(W * 0.5, H * 0.5, H * 0.15, W * 0.5, H * 0.5, H * 0.75);
        vignette.addColorStop(0.0, 'rgba(0,0,0,0.00)');
        vignette.addColorStop(1.0, 'rgba(0,0,0,0.42)');
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, W, H);

        // In-pipe probe: show local speed/pressure only when the pointer is inside the pipe.
        if (pointer) {{
          const x01 = (pointer.x - pipeX) / pipeW;
          if (x01 >= 0.0 && x01 <= 1.0) {{
            const yCenter = pipeY - hAtX01_m(x01) * hScale;
            const rPx = radiusAtX01(x01) * pxPerM;
            const inside = Math.abs(pointer.y - yCenter) <= rPx;
            if (inside) {{
              const v = velAtX01(x01);
              const p = pressureAtX01_kPa(x01);
              const dCm = radiusAtX01(x01) * 2.0 * 100.0;

              ctx.beginPath();
              ctx.arc(pointer.x, pointer.y, 4, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
              ctx.fill();

              const cardW = 210;
              const cardH = 92;
              let tx = pointer.x + 14;
              let ty = pointer.y - cardH - 14;
              if (tx + cardW > W - 10) tx = pointer.x - cardW - 14;
              if (ty < 10) ty = pointer.y + 14;

              drawRoundedRect(tx, ty, cardW, cardH, 12, 'rgba(2, 6, 23, 0.78)', 'rgba(148, 163, 184, 0.22)');

              ctx.fillStyle = 'rgba(226, 232, 240, 0.92)';
              ctx.font = '800 13px system-ui, -apple-system, Segoe UI, Roboto, Arial';
              ctx.fillText('Local readout', tx + 12, ty + 22);

              ctx.fillStyle = 'rgba(226, 232, 240, 0.78)';
              ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
              ctx.fillText(`Speed: ${{v.toFixed(3)}} m/s`, tx + 12, ty + 42);
              ctx.fillText(`Pressure: ${{p.toFixed(3)}} kPa`, tx + 12, ty + 60);
              ctx.fillText(`Diameter: ${{dCm.toFixed(2)}} cm`, tx + 12, ty + 78);
              ctx.fillText(`x: ${{x01.toFixed(3)}}`, tx + 150, ty + 22);
            }}
          }}
        }}

        requestAnimationFrame(render);
      }}

      requestAnimationFrame(render);
    </script>
  </body>
</html>
"""

# ----------------------------
# Streamlit app layout and UI
# ----------------------------


def init_session_state() -> None:
    """Initialize Streamlit session state for storing trial data."""
    # Human-readable column names (with units) for the Data Log table and CSV export.
    professional_columns = [
        "Trial",
        "Density (kg/m³)",
        "Flow rate (L/s)",
        "Left diameter (cm)",
        "Middle diameter (cm)",
        "Right diameter (cm)",
        "Left height (m)",
        "Right height (m)",
        "Inlet pressure (kPa)",
        "Velocity (left) (m/s)",
        "Velocity (middle) (m/s)",
        "Velocity (right) (m/s)",
        "Mass flow rate (kg/s)",
        "Pressure (middle) (kPa)",
        "Pressure (right) (kPa)",
        "Δp (middle - inlet) (kPa)",
        "Δp (right - inlet) (kPa)",
        "Continuity residual (left–middle)",
        "Continuity residual (left–right)",
    ]

    legacy_to_professional = {
        "rho_kg_m3": "Density (kg/m³)",
        "Q_L_s": "Flow rate (L/s)",
        "d_left_cm": "Left diameter (cm)",
        "d_mid_cm": "Middle diameter (cm)",
        "d_right_cm": "Right diameter (cm)",
        "h_left_m": "Left height (m)",
        "h_right_m": "Right height (m)",
        "p_left_kPa": "Inlet pressure (kPa)",
        "v_left_m_s": "Velocity (left) (m/s)",
        "v_mid_m_s": "Velocity (middle) (m/s)",
        "v_right_m_s": "Velocity (right) (m/s)",
        "m_dot_kg_s": "Mass flow rate (kg/s)",
        "p_mid_kPa": "Pressure (middle) (kPa)",
        "p_right_kPa": "Pressure (right) (kPa)",
        "dp_mid_kPa": "Δp (middle - inlet) (kPa)",
        "dp_right_kPa": "Δp (right - inlet) (kPa)",
        "continuity_residual_left_mid": "Continuity residual (left–middle)",
        "continuity_residual_left_right": "Continuity residual (left–right)",
    }

    # First run: create the empty trials table with the professional headers.
    if "trials" not in st.session_state:
        st.session_state.trials = pd.DataFrame(columns=professional_columns)
        return

    # If an existing session has older variable-style headers, migrate them in-place.
    df = st.session_state.trials
    if any(col in df.columns for col in legacy_to_professional):
        df = df.rename(columns=legacy_to_professional)

    # Ensure any missing columns are present (keeps the table stable across updates).
    for col in professional_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep professional columns first; preserve any extra columns at the end.
    ordered_cols = [c for c in professional_columns if c in df.columns] + [
        c for c in df.columns if c not in professional_columns
    ]
    st.session_state.trials = df[ordered_cols]


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Fluids Flow Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # App-wide styling (dark, clean lab-style UI).
    st.markdown(
        """
<style>
.stApp {
  background: radial-gradient(1200px 700px at 10% 0%, rgba(59,130,246,0.18), transparent 55%),
              radial-gradient(900px 650px at 90% 10%, rgba(16,185,129,0.12), transparent 60%),
              linear-gradient(180deg, #0b1220 0%, #070b14 100%);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(2,6,23,0.92));
  border-right: 1px solid rgba(148,163,184,0.15);
}
section[data-testid="stSidebar"] * {
  color: rgba(226,232,240,0.95);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(148,163,184,0.14);
  background: rgba(2,6,23,0.35);
  border-radius: 16px;
}
button[kind="primary"] {
  border-radius: 12px !important;
}
header, footer { visibility: hidden; height: 0px; }
</style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for data logging
    init_session_state()

    # Header: title + concept badges.
    col_title, col_badges = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with col_title:
        st.markdown(
            """
<div style="padding: 6px 0 2px 0;">
  <div style="font-size: 34px; font-weight: 800; letter-spacing: -0.02em; color: rgba(248,250,252,0.96);">
    Fluids Flow Lab
  </div>
  <div style="margin-top: 6px; font-size: 15px; color: rgba(226,232,240,0.78);">
    Visualize continuity and Bernoulli: dots accelerate through constrictions and pressure shifts with elevation.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col_badges:
        st.markdown(
            """
<div style="display:flex; gap:8px; justify-content:flex-end; padding: 8px 0;">
  <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.18); background: rgba(15,23,42,0.55); color: rgba(226,232,240,0.92); font-size: 12px;">
    Continuity
  </span>
  <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.18); background: rgba(15,23,42,0.55); color: rgba(226,232,240,0.92); font-size: 12px;">
    Bernoulli
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("### Controls")

    with st.sidebar.expander("Flow & Fluid", expanded=True):
        # Primary physical inputs: flow rate and density.
        Q_L_s = st.slider(
            "Flow rate Q (L/s)",
            min_value=0.5,
            max_value=50.0,
            value=10.0,
            step=0.5,
        )
        rho = st.slider(
            "Density ρ (kg/m³)",
            min_value=500.0,
            max_value=2000.0,
            value=1000.0,
            step=10.0,
        )
        g = st.slider(
            "Gravity g (m/s²)",
            min_value=0.0,
            max_value=20.0,
            value=9.81,
            step=0.01,
        )

    # Fixed inlet pressure used as the reference for the Bernoulli pressure field.
    p1_kPa = 200.0

    with st.sidebar.expander("Pipe Geometry", expanded=True):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            d_left_cm = st.slider(
                "Left diameter (cm)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
            )
            d_mid_cm = st.slider(
                "Middle diameter (cm)",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.5,
            )
        with col_d2:
            d_right_cm = st.slider(
                "Right diameter (cm)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
            )
            h_left_m = st.slider(
                "Left height (m)",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
            )
            h_right_m = st.slider(
                "Right height (m)",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
            )

    # Convert UI units to SI units for physics calculations.
    Q_m3_s = Q_L_s / 1000.0  # L/s → m³/s
    d_left_m = d_left_cm / 100.0  # cm → m
    d_mid_m = d_mid_cm / 100.0  # cm → m
    d_right_m = d_right_cm / 100.0  # cm → m
    p_left_pa = p1_kPa * 1000.0  # kPa → Pa

    # Cross-sectional areas and velocities from continuity (Q = A v).
    A_left = circle_area_from_diameter(d_left_m)
    A_mid = circle_area_from_diameter(d_mid_m)
    A_right = circle_area_from_diameter(d_right_m)

    v_left = Q_m3_s / A_left
    v_mid = Q_m3_s / A_mid
    v_right = Q_m3_s / A_right

    m_dot = rho * Q_m3_s

    h_mid_m = (h_left_m + h_right_m) / 2.0
    p_mid_pa = p_left_pa + 0.5 * rho * (v_left**2 - v_mid**2) + rho * g * (
        h_left_m - h_mid_m
    )
    p_right_pa = p_left_pa + 0.5 * rho * (v_left**2 - v_right**2) + rho * g * (
        h_left_m - h_right_m
    )

    # Numerical check for mass conservation (should be ~0 in the ideal model).
    continuity_residual_left_mid = abs(A_left * v_left - A_mid * v_mid)
    continuity_residual_left_right = abs(A_left * v_left - A_right * v_right)

    with st.sidebar.expander("Visualization", expanded=True):
        paused = st.toggle("Pause", value=False)
        show_dots = st.toggle("Water dots", value=True)
        show_streaks = st.toggle("Water streaks", value=False)
    pipe_html = build_pipe_2d_html(
        rho=rho,
        Q_L_s=Q_L_s,
        d_left_cm=d_left_cm,
        d_mid_cm=d_mid_cm,
        d_right_cm=d_right_cm,
        p1_kPa=p1_kPa,
        g=g,
        h_left_m=h_left_m,
        h_right_m=h_right_m,
        paused=paused,
        show_dots=show_dots,
        show_streaks=show_streaks,
    )

    tab_sim, tab_data, tab_explain = st.tabs(["Simulation", "Data Log", "Explanation"])

    with tab_sim:
        # The live visualization is rendered as a self-contained HTML canvas.
        components.html(pipe_html, height=540)

    with tab_explain:
        st.markdown(
            """
## Overview

This visualization is an idealized, steady-flow model that links what you see (particle motion) to two core relationships:

- **Continuity (incompressible flow):** $Q = A\\,v$
- **Bernoulli (along a streamline, ideal flow):** $p + \\tfrac{1}{2}\\rho v^2 + \\rho g h = \\text{constant}$

In the constricted region, cross-sectional area $A$ decreases, so speed $v$ increases to keep the same flow rate $Q$. As speed increases, static pressure tends to decrease (Bernoulli), with elevation $h$ also contributing through the $\\rho g h$ term.

## How to use the simulation

- Use the **diameter** controls (left / middle / right) to change the pipe area profile.
- Use the **left/right height** controls to slope the pipe and introduce an elevation effect.
- Toggle **Water dots** (and optional streaks) to visualize flow motion.
- Move your **cursor (or touch)** inside the pipe to read **local speed** and **local pressure** at that point.

## What to look for

- **Narrower section → faster particles:** continuity ($Q=A v$) forces the velocity to rise where the pipe narrows.
- **Faster region → lower pressure:** Bernoulli predicts a drop in static pressure where velocity is higher (all else equal).
- **Raising the outlet changes pressure:** increasing elevation requires pressure to do more “lifting work” through $\\rho g h$.

## Interactive probe (cursor/touch)

The in-pipe tooltip reports the model’s local estimates:

- **Speed $v(x)$** from continuity using the local diameter.
- **Pressure $p(x)$** from Bernoulli including elevation.

This is intended as a clean, visual way to connect geometry → speed → pressure without needing a separate readout panel.

## Modeling assumptions

- Ideal, incompressible, steady flow (no viscosity/turbulence losses).
- Values are best used to compare **trends** rather than to design real systems.
            """
        )

    with tab_data:
        col_log, col_export = st.columns(2)

        with col_log:
            if st.button("Add current configuration as trial", type="primary"):
                df = st.session_state.trials
                new_index = len(df) + 1

                new_row = pd.DataFrame(
                    {
                        "Trial": [new_index],
                        "Density (kg/m³)": [rho],
                        "Flow rate (L/s)": [Q_L_s],
                        "Left diameter (cm)": [d_left_cm],
                        "Middle diameter (cm)": [d_mid_cm],
                        "Right diameter (cm)": [d_right_cm],
                        "Left height (m)": [h_left_m],
                        "Right height (m)": [h_right_m],
                        "Inlet pressure (kPa)": [p1_kPa],
                        "Velocity (left) (m/s)": [v_left],
                        "Velocity (middle) (m/s)": [v_mid],
                        "Velocity (right) (m/s)": [v_right],
                        "Mass flow rate (kg/s)": [m_dot],
                        "Pressure (middle) (kPa)": [p_mid_pa / 1000.0],
                        "Pressure (right) (kPa)": [p_right_pa / 1000.0],
                        "Δp (middle - inlet) (kPa)": [
                            (p_mid_pa - p_left_pa) / 1000.0
                        ],
                        "Δp (right - inlet) (kPa)": [
                            (p_right_pa - p_left_pa) / 1000.0
                        ],
                        "Continuity residual (left–middle)": [
                            continuity_residual_left_mid
                        ],
                        "Continuity residual (left–right)": [
                            continuity_residual_left_right
                        ],
                    }
                )

                if df.empty:
                    st.session_state.trials = new_row
                else:
                    st.session_state.trials = pd.concat(
                        [df, new_row], ignore_index=True
                    )
                st.success(f"Trial {new_index} recorded.")

        with col_export:
            if not st.session_state.trials.empty:
                csv_bytes = st.session_state.trials.to_csv(index=False).encode(
                    "utf-8"
                )
                st.download_button(
                    label="Download trials as CSV",
                    data=csv_bytes,
                    file_name="fluids_flow_lab_trials.csv",
                    mime="text/csv",
                )
            else:
                st.info("No trials recorded yet. Add at least one trial first.")

        if not st.session_state.trials.empty:
            st.dataframe(st.session_state.trials, width="stretch")

    # Short note about model limitations and safety
    st.markdown(
        """
---
### Model assumptions and safety notes

- This is an **idealized** model: it neglects viscosity, turbulence, and energy losses.
- Use it to build intuition and to compare trends, not to design real hardware.
- Classroom safety: when pairing this with physical demonstrations, always follow your
  institution's safety rules for pressurized systems and liquids.
        """
    )


if __name__ == "__main__":
    # Run the Streamlit app when executed as a script.
    # To start the app from a terminal, run:
    #   streamlit run app.py
    main()
