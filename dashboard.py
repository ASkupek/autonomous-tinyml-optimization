"""Autonomous Driving Control Center - Interactive WebGUI Dashboard.

Integrated interface providing live pipeline execution streaming, interactive
multi-objective Pareto analysis, dynamic loss parsing, and real-time profile data updating.
"""

import asyncio
import glob
import json
import os
import platform
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional

from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Root directory calibration
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
EXPORTED_MODELS_DIR: str = os.path.join(PROJECT_ROOT, "ml_pipeline", "exported_models")

# Global reference for sub-process management
current_process: Optional[subprocess.Popen] = None


def get_available_json_files() -> Dict[str, str]:
    """Finds all model metadata JSON profiles in the exported_models directory.

    Returns:
        Dict[str, str]: Mapping of file names to their full absolute paths.
    """
    if not os.path.exists(path=EXPORTED_MODELS_DIR):
        return {}

    files = glob.glob(pathname=os.path.join(EXPORTED_MODELS_DIR, "model_rank_*_profile.json"))
    return {os.path.basename(p=f): f for f in sorted(files)}


def parse_model_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads and parses a selected JSON profile file.

    Args:
        file_path (str): Path to target JSON file.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON content or None if parsing fails.
    """
    if not file_path or not os.path.exists(path=file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(fp=f)
    except Exception as e:
        print(f"[GUI Error] Could not parse {file_path}: {e}")
        return None


def extract_loss(profile: Dict[str, Any]) -> float:
    """Extracts evaluation loss checking multiple potential key names."""
    for key in ("validation_loss", "test_loss", "loss", "mse", "val_mse"):
        if key in profile and profile[key] is not None:
            return float(profile[key])
    return 0.0


def build_advanced_pareto_graph(profiles: List[Dict[str, Any]]) -> go.Figure:
    """Builds an interactive Bubble Chart Pareto Front graph."""
    fig = go.Figure()

    if not profiles:
        fig.add_annotation(
            text="No exported model profiles loaded.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )
    else:
        activation_colors = {
            "relu": "#38bdf8",
            "tanh": "#f59e0b",
            "leaky_relu": "#10b981",
            "gelu": "#a855f7"
        }

        x_vals = [p.get("tflite_size_kb", p.get("model_size_kb", 0)) for p in profiles]
        y_vals = [extract_loss(profile=p) for p in profiles]
        ranks = [p.get("rank", 0) for p in profiles]
        param_counts = [p.get("parameter_count", 1000) for p in profiles]
        activations = [str(p.get("activation_type", "relu")).lower() for p in profiles]
        layer_structures = [str(p.get("layer_structure", [])) for p in profiles]

        min_p, max_p = (min(param_counts), max(param_counts)) if param_counts else (1, 1000)
        bubble_sizes = [
            12 + (p - min_p) / (max_p - min_p + 1e-5) * 24 for p in param_counts
        ]

        marker_colors = [activation_colors.get(act, "#38bdf8") for act in activations]

        hover_texts = [
            f"<b>Rank #{r}</b><br>"
            f"Layers: {layers}<br>"
            f"Activation: {act.upper()}<br>"
            f"INT8 Size: {x} KB<br>"
            f"Loss (MSE): {y:.6f}<br>"
            f"Params: {params:,}"
            for r, layers, act, x, y, params in zip(
                ranks, layer_structures, activations, x_vals, y_vals, param_counts
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=[f"#{r}" for r in ranks],
                textposition="top center",
                hoverinfo="text",
                hovertext=hover_texts,
                marker=dict(
                    size=bubble_sizes,
                    color=marker_colors,
                    opacity=0.85,
                    line=dict(width=1.5, color="#f8fafc"),
                ),
            )
        )

        sorted_indices = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
        line_x = [x_vals[i] for i in sorted_indices]
        line_y = [y_vals[i] for i in sorted_indices]

        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Pareto Frontier",
                line=dict(color="#0284c7", width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text="Pareto Trade-off: Validation Loss vs INT8 Size (Bubble Size = Params)",
            font=dict(size=13, color="#cbd5e1"),
        ),
        xaxis=dict(title="TFLite INT8 Size (KB)", gridcolor="#1e293b"),
        yaxis=dict(title="Validation Loss (MSE)", gridcolor="#1e293b", tickformat=".6f"),
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        showlegend=False,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        height=380,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def build_size_and_error_comparison_graph(profiles: List[Dict[str, Any]]) -> go.Figure:
    """Builds side-by-side comparison bar charts for Model Sizes and Quantization Error."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "1. Model Footprint Comparison (Float32 vs INT8 TFLite Size)",
            "2. Quantization Error (TF vs TFLite INT8 MAE)"
        ),
        horizontal_spacing=0.15
    )

    if not profiles:
        fig.add_annotation(text="No profiles loaded.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#94a3b8"), row=1, col=1)
        fig.add_annotation(text="No profiles loaded.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#94a3b8"), row=1, col=2)
    else:
        sorted_profiles = sorted(profiles, key=lambda p: p.get("rank", 99))
        ranks = [f"Rank #{p.get('rank', '?')}" for p in sorted_profiles]
        
        float_sizes = [p.get("model_size_kb", 0.0) for p in sorted_profiles]
        int8_sizes = [p.get("tflite_size_kb", p.get("model_size_kb", 0.0)) for p in sorted_profiles]
        tflite_maes = [p.get("verification", {}).get("tf_tflite_mae", 0.0) for p in sorted_profiles]

        # 1. Float32 size bar (Left subplot, column 1)
        fig.add_trace(
            go.Bar(x=ranks, y=float_sizes, name="Float32 Size (KB)", marker_color="#64748b"),
            row=1, col=1
        )
        # 2. INT8 TFLite size bar (Left subplot, column 1 - grouped side-by-side)
        fig.add_trace(
            go.Bar(x=ranks, y=int8_sizes, name="INT8 Size (KB)", marker_color="#38bdf8"),
            row=1, col=1
        )
        # 3. Quantization Error bar (Right subplot, column 2)
        fig.add_trace(
            go.Bar(x=ranks, y=tflite_maes, name="INT8 MAE", marker_color="#10b981", text=[f"{v:.5f}" for v in tflite_maes], textposition="auto"),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        height=380,
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30)
    )

    fig.update_xaxes(title_text="Model Rank", gridcolor="#1e293b", row=1, col=1)
    fig.update_yaxes(title_text="Size (KB)", gridcolor="#1e293b", row=1, col=1)

    fig.update_xaxes(title_text="Model Rank", gridcolor="#1e293b", row=1, col=2)
    fig.update_yaxes(title_text="Mean Absolute Error (MAE)", gridcolor="#1e293b", tickformat=".5f", row=1, col=2)

    return fig


def build_nas_exploration_graph(history_path: str = None) -> go.Figure:
    """Builds a clean 2-part NAS visualization: Pareto Exploration Space and Convergence Progress."""
    if history_path is None or not os.path.exists(history_path):
        history_files = glob.glob(os.path.join(EXPORTED_MODELS_DIR, "nas_history_*.json"))
        if history_files:
            history_path = max(history_files, key=os.path.getmtime)
        else:
            history_path = os.path.join(EXPORTED_MODELS_DIR, "nas_history_latest.json")

    if not os.path.exists(history_path):
        fig = go.Figure()
        fig.add_annotation(
            text="NAS optimization history not found.<br>Please run 'python main.py --step train'.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8")
        )
        return fig

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[GUI Error] Error reading NAS history {history_path}: {e}")
        data = []

    valid_data = [d for d in data if d.get("val_loss", 999.0) < 10.0]

    if not valid_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid candidates found in history.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8")
        )
        return fig

    gens = [d["gen"] for d in valid_data]
    costs = [d["tinyml_cost"] for d in valid_data]
    losses = [d["val_loss"] for d in valid_data]
    is_pareto = [d.get("is_pareto", False) for d in valid_data]
    candidates = [d.get("candidate_idx", 0) for d in valid_data]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "1. Pareto Search Space (Loss vs TinyML Cost)",
            "2. Optimization Convergence (Best Loss per Generation)"
        ),
        horizontal_spacing=0.15
    )

    non_pareto_x = [costs[i] for i in range(len(valid_data)) if not is_pareto[i]]
    non_pareto_y = [losses[i] for i in range(len(valid_data)) if not is_pareto[i]]
    non_pareto_gen = [gens[i] for i in range(len(valid_data)) if not is_pareto[i]]
    non_pareto_cand = [candidates[i] for i in range(len(valid_data)) if not is_pareto[i]]

    fig.add_trace(
        go.Scatter(
            x=non_pareto_x,
            y=non_pareto_y,
            mode="markers",
            name="Evaluated Models",
            marker=dict(
                size=11,
                color=non_pareto_gen,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Generation", len=0.8, y=0.5, x=0.45),
                opacity=0.85,
                line=dict(width=1, color="#f8fafc")
            ),
            text=[f"Gen: {g}, Candidate: {c}" for g, c in zip(non_pareto_gen, non_pareto_cand)],
            hovertemplate="<b>%{text}</b><br>TinyML Cost: %{x:.2f}<br>Loss: %{y:.5f}<extra></extra>"
        ),
        row=1, col=1
    )

    pareto_x = [costs[i] for i in range(len(valid_data)) if is_pareto[i]]
    pareto_y = [losses[i] for i in range(len(valid_data)) if is_pareto[i]]
    if pareto_x:
        fig.add_trace(
            go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode="markers",
                name="Pareto Optimal",
                marker=dict(size=15, color="#f59e0b", symbol="star"),
                hovertemplate="<b>Pareto Optimum</b><br>TinyML Cost: %{x:.2f}<br>Loss: %{y:.5f}<extra></extra>"
            ),
            row=1, col=1
        )

    unique_gens = sorted(list(set(gens)))
    min_losses = [min([losses[i] for i in range(len(valid_data)) if gens[i] == g]) for g in unique_gens]
    avg_losses = [sum([losses[i] for i in range(len(valid_data)) if gens[i] == g]) / len([losses[i] for i in range(len(valid_data)) if gens[i] == g]) for g in unique_gens]

    fig.add_trace(
        go.Scatter(
            x=unique_gens, y=min_losses,
            mode="lines+markers",
            name="Best Loss (Min)",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=unique_gens, y=avg_losses,
            mode="lines+markers",
            name="Population Average Loss",
            line=dict(color="#38bdf8", width=2, dash="dash"),
            marker=dict(size=6)
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        height=500,
        showlegend=True,
        margin=dict(l=50, r=20, t=60, b=40)
    )

    fig.update_xaxes(title_text="TinyML Cost (Footprint)", gridcolor="#1e293b", row=1, col=1)
    fig.update_yaxes(title_text="Validation Loss (MSE)", gridcolor="#1e293b", tickformat=".4f", row=1, col=1)

    fig.update_xaxes(title_text="Generation", dtick=1, gridcolor="#1e293b", row=1, col=2)
    fig.update_yaxes(title_text="MSE Loss", gridcolor="#1e293b", tickformat=".4f", row=1, col=2)

    return fig


def build_gui() -> None:
    """Renders layout including live execution, Pareto plots, and JSON file viewer."""
    ui.colors(primary="#0284c7", secondary="#475569", accent="#38bdf8")

    # HEADER
    with ui.header().classes("items-center justify-between bg-slate-950 px-8 py-3 border-b border-slate-800"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("precision_manufacturing", size="sm", color="sky")
            ui.label("AUTONOMOUS TINYML CONTROL CENTER").classes("text-md font-bold text-slate-100 font-mono")

        with ui.row().classes("items-center gap-2"):
            ui.icon("memory", size="xs", color="emerald")
            ui.label("NXP MCXN947 HIL Engine").classes("text-xs text-emerald-400 font-mono")

    # TABS NAVIGATION
    with ui.tabs().classes("w-full bg-slate-900 text-slate-400 border-b border-slate-800") as tabs:
        t_run = ui.tab("1. Execution & Live Stream")
        t_pareto = ui.tab("2. Pareto & File Viewer")
        t_convergence = ui.tab("3. NAS Convergence History")

    with ui.tab_panels(tabs, value=t_run).classes("w-full p-6 bg-slate-950 text-slate-200 min-h-screen"):

        # ==========================================
        # TAB 1: EXECUTION & LIVE STREAM
        # ==========================================
        with ui.tab_panel(t_run):
            ui.label("Live Training & Execution Monitor").classes("text-xl font-bold mb-1 text-slate-100")
            ui.label("Trigger pipeline steps (main.py) with real-time process output streaming.").classes("text-xs text-slate-400 mb-6")

            with ui.row().classes("w-full gap-6 items-start mb-6"):
                # Control Card
                with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 w-96"):
                    ui.label("Execution Control").classes("text-sm font-bold mb-4 text-sky-400 uppercase font-mono")

                    step_select = ui.select(
                        options={"train": "1. NAS & Model Training", "get-data": "2. Data Ingestion (CARLA)", "test": "3. MCU HIL Verification"},
                        value="train",
                        label="Pipeline Step (--step)"
                    ).classes("w-full bg-slate-950 text-xs mb-4")

                    closed_loop_chk = ui.checkbox("Run Closed-Loop in CARLA (--closed-loop)", value=False).classes("text-xs text-slate-300 mb-4")

                    status_lbl = ui.label("Status: Idle").classes("text-xs font-mono text-slate-300 mb-4")
                    spinner = ui.spinner(size="md").classes("self-center hidden mb-4")

                    run_btn = ui.button("START PROCESS", color="sky").classes("w-full py-3 font-bold text-xs tracking-wide mb-2")
                    stop_btn = ui.button("ABORT PROCESS", color="red").classes("w-full py-2 font-bold text-xs tracking-wide hidden")

                # Log Box Card
                with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 flex-1"):
                    ui.label("Console Output Stream (STDOUT / STDERR)").classes("text-sm font-bold mb-3 text-sky-400 font-mono")
                    log_box = ui.log(max_lines=400).classes("w-full h-96 bg-slate-950 p-3 rounded border border-slate-800 font-mono text-xs text-sky-300")

            async def start_pipeline() -> None:
                global current_process
                run_btn.disable()
                run_btn.classes(add="hidden")
                stop_btn.classes(remove="hidden")
                spinner.classes(remove="hidden")

                selected_step = step_select.value
                is_closed_loop = closed_loop_chk.value

                status_lbl.set_text(f"Status: Executing '{selected_step}'...")
                log_box.clear()
                log_box.push(f"[INFO] Launching process: main.py --step {selected_step}\n")

                cmd = [sys.executable, "-u", "main.py", "--step", str(selected_step)]
                if selected_step == "test" and is_closed_loop:
                    cmd.append("--closed-loop")

                try:
                    current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=PROJECT_ROOT,
                    )

                    def read_output() -> None:
                        if current_process and current_process.stdout:
                            for line in iter(current_process.stdout.readline, ""):
                                if line:
                                    log_box.push(line)
                            current_process.stdout.close()

                    await asyncio.to_thread(read_output)
                    await asyncio.to_thread(current_process.wait)

                    if current_process.returncode == 0:
                        status_lbl.set_text("Status: Execution finished successfully!")
                    else:
                        status_lbl.set_text(f"Status: Process terminated (Code {current_process.returncode})")

                except Exception as ex:
                    log_box.push(traceback.format_exc())
                    log_box.push(f"\n[ERROR] Pipeline execution failed: {ex}\n")
                    status_lbl.set_text("Status: Execution Error")
                finally:
                    current_process = None
                    spinner.classes(add="hidden")
                    stop_btn.classes(add="hidden")
                    run_btn.classes(remove="hidden")
                    run_btn.enable()

            async def stop_pipeline() -> None:
                global current_process
                if current_process and current_process.poll() is None:
                    log_box.push("\n[WARN] Aborting process by user request...\n")
                    status_lbl.set_text("Status: Aborting...")
                    try:
                        if platform.system() == "Windows":
                            subprocess.call(["taskkill", "/F", "/T", "/PID", str(current_process.pid)])
                        else:
                            current_process.terminate()
                    except Exception as e:
                        log_box.push(f"[ERROR] Failed to terminate process: {e}\n")

            run_btn.on_click(start_pipeline)
            stop_btn.on_click(stop_pipeline)

        # ==========================================
        # TAB 2: PARETO & FILE VIEWER (Z DUAL-COMPARISON GRAFOM)
        # ==========================================
        with ui.tab_panel(t_pareto):
            ui.label("Inspect Pareto Models & Exported JSON Profiles").classes("text-xl font-bold mb-1 text-slate-100")
            ui.label("Select an exported file from ml_pipeline/exported_models/ to view exact architecture parameters.").classes("text-xs text-slate-400 mb-6")

            file_map = get_available_json_files()
            all_profiles = [parse_model_json(f) for f in file_map.values() if parse_model_json(f)]

            with ui.row().classes("w-full gap-6 items-start mb-6"):
                # Pareto Plot Card
                with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 flex-1"):
                    ui.label("Pareto Trade-off Chart").classes("text-sm font-bold mb-2 text-sky-400 font-mono")
                    ui.plotly(build_advanced_pareto_graph(profiles=all_profiles)).classes("w-full rounded-lg overflow-hidden")

                # Profile Details Card
                with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 w-96"):
                    ui.label("Profile Data Inspector").classes("text-sm font-bold mb-4 text-sky-400 font-mono")
                    
                    if file_map:
                        file_select = ui.select(
                            options=list(file_map.keys()),
                            value=list(file_map.keys())[0]
                        ).classes("w-full bg-slate-950 text-xs mb-4")

                        details_container = ui.column().classes("gap-2 font-mono text-xs w-full")

                        def render_selected_details(file_name: str) -> None:
                            details_container.clear()
                            selected_profile = parse_model_json(file_map[file_name])
                            if selected_profile:
                                loss_val = extract_loss(selected_profile)
                                tinyml_cost = selected_profile.get("tinyml_cost", selected_profile.get("model_size_kb", "N/A"))
                                tflite_size = selected_profile.get("tflite_size_kb", "N/A")
                                
                                with details_container:
                                    ui.label(f"Rank: #{selected_profile.get('rank', 'N/A')}").classes("text-sky-300 font-bold")
                                    ui.label(f"Layers: {selected_profile.get('layer_structure', [])}").classes("text-emerald-400")
                                    ui.label(f"Activation: {str(selected_profile.get('activation_type', '')).upper()}").classes("text-amber-300")
                                    ui.label(f"Params Count: {selected_profile.get('parameter_count', 0):,}").classes("text-slate-200")
                                    ui.label(f"Float32 Size: {selected_profile.get('model_size_kb', 0)} KB").classes("text-slate-200")
                                    ui.label(f"INT8 TFLite Size: {tflite_size} KB").classes("text-emerald-300 font-bold")
                                    ui.label(f"TinyML Cost: {tinyml_cost}").classes("text-purple-400 font-bold")
                                    ui.label(f"Loss (MSE): {loss_val:.6f}").classes("text-sky-300 font-bold")

                        render_selected_details(file_select.value)
                        file_select.on_value_change(lambda e: render_selected_details(e.value))
                    else:
                        ui.label("No exported model profiles found.").classes("text-xs text-amber-400 font-mono")

            with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 w-full"):
                ui.plotly(build_size_and_error_comparison_graph(profiles=all_profiles)).classes("w-full rounded-lg overflow-hidden")

        # ==========================================
        # TAB 3: NAS CONVERGENCE HISTORY
        # ==========================================
        with ui.tab_panel(t_convergence):
            ui.label("NAS Evolutionary Search Exploration & Convergence").classes("text-xl font-bold mb-1 text-slate-100")
            ui.label("Multi-generation exploration history and convergence tracking for academic reporting.").classes("text-xs text-slate-400 mb-6")

            with ui.card().classes("bg-slate-900 p-5 rounded-lg border border-slate-800 w-full"):
                ui.plotly(build_nas_exploration_graph()).classes("w-full rounded-lg overflow-hidden")


if __name__ in {"__main__", "__mp_main__"}:
    ui.page_title("Autonomous TinyML AI Dashboard")
    build_gui()
    ui.run(title="TinyML Analytics Engine", port=8081, reload=False)