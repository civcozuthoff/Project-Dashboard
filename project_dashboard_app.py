
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

st.set_page_config(page_title="Project Portfolio Dashboard", layout="wide")

DEFAULT_EXCEL = "project_dashboard_export.xlsx"

# ---------- Helpers ----------
ns = {"msproj": "http://schemas.microsoft.com/project"}

def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def parse_project_xml(file_like):
    try:
        tree = ET.parse(file_like)
        root = tree.getroot()
    except Exception as e:
        st.error(f"Failed to parse XML: {e}")
        return None, None

    def gettext(path):
        el = root.find(path, ns)
        return el.text if el is not None and el.text is not None else None

    name = gettext("msproj:Name") or "Unnamed Project"
    title = gettext("msproj:Title")
    start_dt = parse_dt(gettext("msproj:StartDate"))
    finish_dt = parse_dt(gettext("msproj:FinishDate"))
    last_saved = parse_dt(gettext("msproj:LastSaved"))
    current_dt = parse_dt(gettext("msproj:CurrentDate")) or datetime.now()

    tasks = []
    for t in root.findall("msproj:Tasks/msproj:Task", ns):
        uid_el = t.find("msproj:UID", ns)
        if uid_el is not None and uid_el.text == "0":
            continue

        def tg(tag, default=None):
            el = t.find(f"msproj:{tag}", ns)
            return el.text if el is not None and el.text is not None else default

        summary = tg("Summary", "0") == "1"
        percent = float(tg("PercentComplete", "0") or 0)
        active = (tg("Active", "1") == "1")
        start_task = parse_dt(tg("Start"))
        finish_task = parse_dt(tg("Finish"))
        critical = tg("Critical", "0") == "1"
        milestone = tg("Milestone", "0") == "1"
        notes = tg("Notes", "")
        res = tg("ResourceNames", None)

        is_open = (not summary) and active and (percent < 100.0)

        tasks.append({
            "Project": name,
            "TaskUID": tg("UID",""),
            "TaskName": tg("Name",""),
            "IsSummary": summary,
            "IsMilestone": milestone,
            "PercentComplete": percent,
            "Active": active,
            "Critical": critical,
            "Start": start_task,
            "Finish": finish_task,
            "Notes": notes,
            "ResourceNames": res,
            "IsOpenAction": is_open,
        })

    df_tasks = pd.DataFrame(tasks)
    non_summary = df_tasks[~df_tasks["IsSummary"]] if not df_tasks.empty else df_tasks
    open_actions = non_summary[non_summary["IsOpenAction"]] if not df_tasks.empty else df_tasks
    overall_pct = round(non_summary["PercentComplete"].mean(), 1) if not non_summary.empty else 0.0
    late_open = open_actions[open_actions["Finish"].notna() & (open_actions["Finish"] < current_dt)]
    crit_open = open_actions[open_actions["Critical"] == True]

    summary = {
        "ProjectName": name,
        "ProjectTitle": title,
        "ProjectFile": getattr(file_like, "name", name),
        "PlannedStart": start_dt,
        "PlannedFinish": finish_dt,
        "LastSaved": last_saved,
        "OverallPercentComplete": overall_pct,
        "TotalTasks": int(len(non_summary)) if not df_tasks.empty else 0,
        "OpenActions": int(len(open_actions)),
        "CriticalOpenActions": int(len(crit_open)),
        "LateOpenActions": int(len(late_open)),
    }
    return summary, open_actions

def classify_status(row, today):
    if row.get("LateOpenActions", 0) > 0:
        return "Late"
    days_to_finish = None
    if pd.notna(row.get("PlannedFinish")):
        try:
            days_to_finish = (pd.to_datetime(row["PlannedFinish"]) - today).days
        except Exception:
            pass
    if (row.get("OverallPercentComplete", 0) < 50) or (days_to_finish is not None and days_to_finish <= 14 and row.get("OpenActions", 0) > 0):
        return "At Risk"
    return "On Track"

STATUS_COLOR = {"On Track":"#2E7D32","At Risk":"#EF6C00","Late":"#C62828"}

def kpi_tile(label, value, help_text=None):
    c = st.container()
    c.metric(label, value)
    if help_text:
        c.caption(help_text)

def burndown_plot(df, xcol, ycol, title):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df[xcol], df[ycol], marker='o')
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True)
    st.pyplot(fig)

# ---------- Data Sources ----------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose data source:", ["Use existing Excel", "Upload XML files"])

if mode == "Use existing Excel":
    excel_file = st.sidebar.text_input("Path to Excel", DEFAULT_EXCEL)
    if not os.path.exists(excel_file):
       st.error(f"Excel not found: {excel_file}. Upload XMLs instead or place the file next to the app.")
    st.stop()
    df_summary = pd.read_excel(excel_file, sheet_name="Summary")
    try:
        df_all_open = pd.read_excel(excel_file, sheet_name="AllOpenActions")
    except Exception:
        st.warning("AllOpenActions sheet not found; some visuals may be limited.")
        df_all_open = pd.DataFrame()
else:
    uploads = st.sidebar.file_uploader("Upload one or more MS Project XML files", type=["xml"], accept_multiple_files=True)
    summaries = []
    open_frames = []
    if uploads:
        for up in uploads:
            summary, open_df = parse_project_xml(up)
            if summary is not None:
                summaries.append(summary)
            if open_df is not None and not open_df.empty:
                open_frames.append(open_df)
        df_summary = pd.DataFrame(summaries)
        df_all_open = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame()
    else:
        st.info("Upload XMLs to build the dashboard.")
        st.stop()

st.title("Project Portfolio Dashboard")

if df_summary.empty:
    st.warning("No summary data available.")
    st.stop()

today = datetime.now()

# Status classification
df_summary = df_summary.copy()
df_summary["Status"] = df_summary.apply(lambda r: classify_status(r, today), axis=1)
df_summary["StatusColor"] = df_summary["Status"].map(STATUS_COLOR)

# KPI row
col1, col2, col3, col4 = st.columns(4)
kpi_tile("Total Projects", len(df_summary))
kpi_tile("Total Open Actions", int(df_summary["OpenActions"].sum()))
kpi_tile("Critical Open Actions", int(df_summary["CriticalOpenActions"].sum()))
kpi_tile("Late Open Actions", int(df_summary["LateOpenActions"].sum()))

# Status tiles table
st.subheader("Project Status")
st.dataframe(df_summary[["ProjectName","OverallPercentComplete","OpenActions","CriticalOpenActions","LateOpenActions","PlannedFinish","Status"]])

# Filters
st.sidebar.header("Filters")
status_filter = st.sidebar.multiselect("Status", options=df_summary["Status"].unique().tolist(), default=df_summary["Status"].unique().tolist())
proj_filter = st.sidebar.multiselect("Projects", options=df_summary["ProjectName"].tolist(), default=df_summary["ProjectName"].tolist())

df_summary_f = df_summary[df_summary["Status"].isin(status_filter) & df_summary["ProjectName"].isin(proj_filter)]

# Open actions source
if df_all_open is None or df_all_open.empty:
    st.info("No open actions found.")
else:
    # Filter to selected projects
    df_open_f = df_all_open[df_all_open["Project"].isin(df_summary_f["ProjectName"])].copy()
    # Owner load
    if "ResourceNames" in df_open_f.columns:
        df_open_f["Owner"] = df_open_f["ResourceNames"].fillna("Unassigned")
    else:
        df_open_f["Owner"] = "Unassigned"
    # Aging
    df_open_f["Finish"] = pd.to_datetime(df_open_f["Finish"], errors="coerce")
    df_open_f["DaysLate"] = (today - df_open_f["Finish"]).dt.days
    def bucket(d):
        if pd.isna(d) or d < 0: return "Not Due"
        if d <= 7: return "0–7"
        if d <= 14: return "8–14"
        return ">14"
    df_open_f["AgingBucket"] = df_open_f["DaysLate"].apply(bucket)

    st.subheader("Top Open Actions (filtered)")
    cols_show = ["Project","TaskName","PercentComplete","Critical","Start","Finish","Owner","Notes"]
    st.dataframe(df_open_f.sort_values(by=["Critical","Finish","TaskName"], ascending=[False,True,True])[cols_show].head(50))

    # Burndown Overall
    bd_overall = (
        df_open_f.dropna(subset=["Finish"])
        .assign(FinishDate=lambda d: d["Finish"].dt.date)
        .groupby("FinishDate").size().reset_index(name="OpenTasks")
        .sort_values("FinishDate")
    )
    if not bd_overall.empty:
        burndown_plot(bd_overall, "FinishDate", "OpenTasks", "Open Actions Burndown — Portfolio (filtered)")

    # Burndown by project
    st.subheader("Burndown by Project")
    for prj, grp in (
        df_open_f.dropna(subset=["Finish"])
        .assign(FinishDate=lambda d: d["Finish"].dt.date)
        .groupby(["Project","FinishDate"]).size().reset_index(name="OpenTasks")
        .sort_values(["Project","FinishDate"])
        .groupby("Project")
    ):
        st.markdown(f"**{prj}**")
        burndown_plot(grp, "FinishDate", "OpenTasks", f"Open Actions — {prj}")

    # Owner load chart
    st.subheader("Owner Load (Open Actions by Owner)")
    owner_counts = df_open_f.groupby("Owner")["TaskUID"].count().sort_values(ascending=False).reset_index(name="OpenActions")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(owner_counts["Owner"], owner_counts["OpenActions"])
    ax.set_title("Open Actions by Owner")
    ax.set_xlabel("Owner")
    ax.set_ylabel("Open Actions")
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    st.pyplot(fig)

    # Aging buckets stacked by project
    st.subheader("Aging Buckets (by Project)")
    age_tbl = (
        df_open_f[df_open_f["AgingBucket"] != "Not Due"]
        .groupby(["Project","AgingBucket"]).size().reset_index(name="Tasks")
        .pivot(index="Project", columns="AgingBucket", values="Tasks").fillna(0)
        .reindex(columns=["0–7","8–14",">14"], fill_value=0)
    )
    if not age_tbl.empty:
        fig2, ax2 = plt.subplots(figsize=(8,4))
        bottom = None
        for col in age_tbl.columns:
            vals = age_tbl[col].values
            ax2.bar(age_tbl.index, vals, bottom=bottom, label=col)
            bottom = vals if bottom is None else bottom + vals
        ax2.set_title("Open Action Aging by Project")
        ax2.set_xlabel("Project")
        ax2.set_ylabel("Tasks")
        ax2.legend(title="Days Late")
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        st.pyplot(fig2)

# Export section
st.sidebar.header("Export")
if st.sidebar.button("Export current view to Excel"):
    # Build a compact export
    out = pd.ExcelWriter("dashboard_export_view.xlsx", engine="xlsxwriter")
    try:
        df_summary_f.to_excel(out, index=False, sheet_name="SummaryFiltered")
        if not (df_all_open is None or df_all_open.empty):
            df_open_f.to_excel(out, index=False, sheet_name="OpenActionsFiltered")
        out.close()
        st.sidebar.success("Saved dashboard_export_view.xlsx")
        with open("dashboard_export_view.xlsx","rb") as f:
            st.sidebar.download_button("Download export", f, file_name="dashboard_export_view.xlsx")
    except Exception as e:
        st.sidebar.error(f"Export failed: {e}")
        try:
            out.close()
        except Exception:
            pass
