import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

df = pd.read_csv("./gemma-lora-power_timeseries_cut.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

df["t_local"] = df["timestamp"].dt.tz_convert("America/Mexico_City")

phase_colors = {
    "dataset": "#1f77b4",
    "load_model": "#ff7f0e",
    "fine_tuning": "#2ca02c",
    "evaluation": "#d62728",
}

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

def plot_signal(ax, col, ylabel, smooth=True):
    ax.plot(df["t_local"], df[col], color="lightgray", linewidth=0.6, alpha=0.7, label="raw")
    if smooth:
        rolling = df[col].rolling(window=30, min_periods=1).mean()
        ax.plot(df["t_local"], rolling, color="#333", linewidth=1.2, label="rolling mean (30)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    for phase, group in df.groupby("phase"):
        ax.axvspan(
            group["t_local"].min(),
            group["t_local"].max(),
            color=phase_colors.get(phase, "gray"),
            alpha=0.08,
            label=f"phase: {phase}",
        )

plot_signal(axes[0], "cpu_e", "CPU power (W)")
plot_signal(axes[1], "ram_e", "RAM power (W)")

ax_e = axes[2]
ax_gpu = ax_e.twinx()
ax_gpu.plot(df["t_local"], df["gpu_e"], color="#9467bd", linewidth=1.0, label="gpu_e")
ax_gpu.set_ylabel("GPU energy (kWh)")

# Single legend per axis, deduplicated
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)

# Show one tick per hour on the bottom axis (change interval=2 for every 2h, etc.),
# with half-hour minor ticks for context.
hour_locator = mdates.HourLocator(interval=1)
minor_locator = mdates.MinuteLocator(byminute=[30])
hour_fmt = mdates.DateFormatter("%H:%M")
for ax in axes:
    ax.xaxis.set_major_locator(hour_locator)
    ax.xaxis.set_major_formatter(hour_fmt)
    ax.xaxis.set_minor_locator(minor_locator)

axes[-1].set_xlabel("Time (America/Mexico_City)")
fig.suptitle("Gemma LoRA — power & energy timeseries", fontsize=13)
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("gemma-lora-power_timeseries.png", dpi=150)
plt.show()
