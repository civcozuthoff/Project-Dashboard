import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime

# ---------- App config & plot style ----------
st.set_page_config(page_title="Project Portfolio Dashboard", layout="wide")
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# ---------- Helpers ----------
ns = {"msproj": "http://schemas.microsoft.com/project"}

def parse_dt(s: str | None):
    if not s:
        return None
    try:
        # MS Project XML can have trailing "Z"
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

def parse_project_xml(file_like):
    """
    Returns:
      summary (dict),
      open_actions (DataFrame),
      all_tasks (DataFrame),
      owner_pairs (TaskUID->Owner DataFrame)
    """
    tree = ET.parse(file_like)
    root = tree.getroot()

    def gettext(path):
        el = root.find(path, ns)
        return el.text if el is not None and el.text is not None else None

    name = gettext("msproj:Name") or getattr(file_like, "name", "Unnamed Project")
    title = gettext("msproj:Title")
    start_dt = parse_dt(gettext("msproj:StartDate"))
    finish_dt = parse_dt(gettext("msproj:FinishDate"))
    last_saved = parse_dt(gettext("msproj:LastSaved"))
    current_dt = parse_dt(gettext("msproj:CurrentDate")) or datetime.now()

    # --- Resources ---
    res_uid_to_name = {}
    for r in root.findall("msproj:Resources/msproj:Resource", ns):
        uid = r.find("msproj:UID", ns)
        nm = r.find("msproj:Name", ns)
        if uid is not None and nm is not None and uid.text:
            res_uid_to_name[uid.text] = nm.text

    # --- Tasks ---
    tasks = []
    for t in root.findall("msproj:Tasks/msproj:Task", ns):
        uid_el = t.find("msproj:UID", ns)
        if uid_el is None or uid_el.text == "0":
            continue

        def tg(tag, default=None):
            el = t.find(f"msproj:{tag}", ns)
            return el.text if el is not None and el.text is not None else default

        task_name = (tg("Name", "") or "").strip()
        if task_name == "":  # EDIT #3: skip tasks without a name
            continue

        summary = (tg("Summary", "0") == "1")
        percent = float(tg("PercentComplete", "0") or 0)
        active = (tg("Active", "1") == "1")
        start_task = parse_dt(tg("Start"))
        finish_task = parse_dt(tg("Finish"))
        actual_finish = parse_dt(tg("ActualFinish"))  # EDIT #2
        critical = (tg("Critical", "0") == "1")
        milestone = (tg("Milestone", "0") == "1")
        notes = tg("Notes", "")
        res_inline = tg("ResourceNames", None)

        preds = []
        for pl in t.findall("msproj:PredecessorLink", ns):
            puid = pl.find("msproj:PredecessorUID", ns)
            if puid is not None and puid.text:
                preds.append(puid.text)

        # EDIT #2: open = not summary, active, <100%, and no ActualFinish recorded
        is_open = (not summary) and active and (percent < 100.0) and (actual_finish is None)

        tasks.append({
            "Project": name,
            "ProjectTitle": title,
            "TaskUID": tg("UID", ""),
            "TaskName": task_name,
            "IsSummary": summary,
            "IsMilestone": milestone,
            "PercentComplete": percent,
            "Active": active,
            "Critical": critical,
            "Start": start_task,
            "Finish": finish_task,
            "ActualFinish": actual_finish,
            "Notes": notes,
            "ResourceNamesInline": res_inline,
            "Predecessors": preds,
            "IsOpenAction": is_open,
        })

    df_tasks = pd.DataFrame(tasks)

    # --- Assignments → Owners map (most reliable) ---
    pairs = []
    for a in root.findall("msproj:Assignments/msproj:Assignment", ns):
        tu = a.find("msproj:TaskUID", ns)
        ru = a.find("msproj:ResourceUID", ns)
        if tu is None or ru is None:
            continue
        t_uid, r_uid = tu.text, ru.text
        if t_uid and (r_uid in res_uid_to_name):
            pairs.append({"TaskUID": t_uid, "Owner": res_uid_to_name[r_uid]})
    df_pairs = pd.DataFrame(pairs)

    # Fallback: split inline ResourceNames (if no assignments)
    if df_pairs.empty and not df_tasks.empty:
        tmp = df_tasks[["TaskUID", "ResourceNamesInline"]].dropna()
        rows = []
        for _, r in tmp.iterrows():
            for part in str(r["ResourceNamesInline"]).replace(";", ",").split(","):
                nm = part.strip()
                if nm:
                    rows.append({"TaskUID": r["TaskUID"], "Owner": nm})
        df_pairs = pd.DataFrame(rows)

    # --- Open, late, crit rollups ---
    non_summary = df_tasks[~df_tasks["IsSummary"]] if not df_tasks.empty else df_tasks
    open_actions = non_summary[non_summary["IsOpenAction"]] if not non_summary.empty else non_summary

    # Late = open and finish < today
    late_open = open_actions.copy()
    late_open = late_open[late_open["Finish"].notna() & (late_open["Finish"] < current_dt)]

    crit_open = open_actions[open_actions["Critical"] == True]
    overall_pct = round(non_summary["PercentComplete"].mean(), 1) if not non_summary.empty else 0.0

    # Bull/Bear dates
    bull_finish = finish_dt
    if not crit_open.empty and crit_open["Finish"].notna().any():
        bear_finish = crit_open["Finish"].max()
    elif not open_actions.empty and open_actions["Finish"].notna().any():
        bear_finish = open_actions["Finish"].max()
    else:
        bear_finish = finish_dt

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
        "BullFinish": bull_finish,
        "BearFinish": bear_finish,
    }
    return summary, open_actions, df_tasks, df_pairs

def kpi(label, value, help_text=None):
    c = st.container()
    c.metric(label, value)
    if help_text: c.caption(help_text)

def small_line(df, xcol, ycol, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(df[xcol], df[ycol], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Data Source ----------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose data source:", ["Upload XML files", "Upload Excel export"])

df_summary = pd.DataFrame()
df_all_open = pd.DataFrame()
df_all_tasks = pd.DataFrame()
df_owner_pairs_all = pd.DataFrame()

if mode == "Upload XML files":
    uploads = st.sidebar.file_uploader(
        "Upload one or more MS Project XML files",
        type=["xml"],
        accept_multiple_files=True
    )
    summaries, open_frames, task_frames, owner_maps = [], [], [], []
    if uploads:
        for up in uploads:
            try:
                summary, open_df, tasks_df, owners_df = parse_project_xml(up)
            except Exception as e:
                st.error(f"Failed to parse {getattr(up, 'name', 'XML')}: {e}")
                continue
            if summary: summaries.append(summary)
            if open_df is not None and not open_df.empty: open_frames.append(open_df)
            if tasks_df is not None and not tasks_df.empty: task_frames.append(tasks_df)
            if owners_df is not None and not owners_df.empty:
                owners_df = owners_df.copy()
                owners_df["Project"] = summary["ProjectName"]
                owner_maps.append(owners_df)
    if summaries:
        df_summary = pd.DataFrame(summaries)
    if open_frames:
        df_all_open = pd.concat(open_frames, ignore_index=True)
    if task_frames:
        df_all_tasks = pd.concat(task_frames, ignore_index=True)
    if owner_maps:
        df_owner_pairs_all = pd.concat(owner_maps, ignore_index=True)

else:
    up_excel = st.sidebar.file_uploader("Upload an Excel export generated by this app", type=["xlsx"])
    if up_excel is not None:
        try:
            df_summary = pd.read_excel(up_excel, sheet_name="Summary")
            df_all_open = pd.read_excel(up_excel, sheet_name="AllOpenActions")
        except Exception as e:
            st.error(f"Could not read Excel: {e}")

if df_summary.empty:
    st.info("Upload XMLs or an Excel export to begin.")
    st.stop()

today = datetime.now()

# ---------- Status classification ----------
def classify_status(row):
    if row.get("LateOpenActions", 0) > 0:
        return "Late"
    days_to_finish = None
    if pd.notna(row.get("PlannedFinish")):
        try:
            days_to_finish = (pd.to_datetime(row["PlannedFinish"]) - today).days
        except Exception:
            pass
    if (row.get("OverallPercentComplete", 0) < 50) or (
        days_to_finish is not None and days_to_finish <= 14 and row.get("OpenActions", 0) > 0
    ):
        return "At Risk"
    return "On Track"

df_summary["Status"] = df_summary.apply(classify_status, axis=1)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Total Projects", len(df_summary))
with c2: kpi("Total Open Actions", int(df_summary["OpenActions"].sum()))
with c3: kpi("Critical Open Actions", int(df_summary["CriticalOpenActions"].sum()))
with c4: kpi("Late Open Actions", int(df_summary["LateOpenActions"].sum()))

# ---------- Status table ----------
st.subheader("Project Status")
st.dataframe(
    df_summary[
        [
            "ProjectName",
            "OverallPercentComplete",
            "OpenActions",
            "CriticalOpenActions",
            "LateOpenActions",
            "PlannedFinish",
            "BullFinish",
            "BearFinish",
            "Status",
        ]
    ]
)

# ---------- Filters ----------
st.sidebar.header("Filters")
status_filter = st.sidebar.multiselect(
    "Status",
    options=df_summary["Status"].unique().tolist(),
    default=df_summary["Status"].unique().tolist(),
)
proj_filter = st.sidebar.multiselect(
    "Projects",
    options=df_summary["ProjectName"].tolist(),
    default=df_summary["ProjectName"].tolist(),
)

df_summary_f = df_summary[
    df_summary["Status"].isin(status_filter) & df_summary["ProjectName"].isin(proj_filter)
]
if df_all_open is not None and not df_all_open.empty:
    df_open_f = df_all_open[df_all_open["Project"].isin(df_summary_f["ProjectName"])].copy()
else:
    df_open_f = pd.DataFrame()

# ---------- 3-across charts ----------
colA, colB, colC = st.columns(3)

# Portfolio Burndown
if not df_open_f.empty:
    bd_overall = (
        df_open_f.dropna(subset=["Finish"])
        .assign(FinishDate=lambda d: pd.to_datetime(d["Finish"]).dt.date)
        .groupby("FinishDate")
        .size()
        .reset_index(name="OpenTasks")
        .sort_values("FinishDate")
    )
    with colA:
        if not bd_overall.empty:
            small_line(bd_overall, "FinishDate", "OpenTasks", "Burndown — Portfolio (filtered)")

# Owner Load (uses Assignments where available)
if not df_open_f.empty:
    if not df_owner_pairs_all.empty:
        owners_map = df_owner_pairs_all[df_owner_pairs_all["Project"].isin(df_summary_f["ProjectName"])][
            ["TaskUID", "Owner"]
        ]
        tmp = df_open_f.merge(owners_map, on="TaskUID", how="left")
        tmp["Owner"] = tmp["Owner"].fillna("Unassigned")
    else:
        tmp = df_open_f.copy()
        tmp["Owner"] = tmp.get("ResourceNamesInline", "Unassigned")
        tmp["Owner"] = tmp["Owner"].fillna("Unassigned")
    owner_counts = (
        tmp.groupby("Owner")["TaskUID"].count().sort_values(ascending=False).reset_index(name="OpenActions")
    )
    with colB:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(owner_counts["Owner"], owner_counts["OpenActions"])
        ax.set_title("Open Actions by Owner")
        ax.set_xlabel("Owner")
        ax.set_ylabel("OpenActions")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

# Aging buckets
if not df_open_f.empty:
    df_open_f["Finish"] = pd.to_datetime(df_open_f["Finish"], errors="coerce")
    df_open_f["DaysLate"] = (today - df_open_f["Finish"]).dt.days

    def bucket(d):
        if pd.isna(d) or d < 0:
            return "Not Due"
        if d <= 7:
            return "0–7"
        if d <= 14:
            return "8–14"
        return ">14"

    df_open_f["AgingBucket"] = df_open_f["DaysLate"].apply(bucket)
    age_tbl = (
        df_open_f[df_open_f["AgingBucket"] != "Not Due"]
        .groupby(["Project", "AgingBucket"])
        .size()
        .reset_index(name="Tasks")
        .pivot(index="Project", columns="AgingBucket", values="Tasks")
        .fillna(0)
        .reindex(columns=["0–7", "8–14", ">14"], fill_value=0)
    )
    with colC:
        if not age_tbl.empty:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            bottom = np.zeros(len(age_tbl))
            for col in age_tbl.columns:
                vals = age_tbl[col].values
                ax2.bar(age_tbl.index, vals, bottom=bottom, label=col)
                bottom = bottom + vals
            ax2.set_title("Aging by Project")
            ax2.set_xlabel("Project")
            ax2.set_ylabel("Tasks")
            ax2.legend(title="Days Late")
            ax2.tick_params(axis="x", rotation=45)
            fig2.tight_layout()
            st.pyplot(fig2)

# ---------- Focus: Late and Upcoming ----------
st.subheader("Focus: Late and Upcoming")

# Top 5 Late Tasks by duration (days late)
if not df_open_f.empty:
    late = df_open_f.copy()
    late["Finish"] = pd.to_datetime(late["Finish"], errors="coerce")
    late = late[late["Finish"].notna() & (late["Finish"] < today)]
    if not late.empty:
        late["DaysLate"] = (today - late["Finish"]).dt.days
        # attach owner when available
        if "Owner" not in late.columns and not df_owner_pairs_all.empty:
            owners_map = df_owner_pairs_all[["TaskUID", "Owner"]]
            late = late.merge(owners_map, on="TaskUID", how="left")
        late["Owner"] = late.get("Owner", late.get("ResourceNamesInline", "Unassigned")).fillna("Unassigned")
        late = late[late["TaskName"].str.len() > 0]  # EDIT #3: ensure names
        top5 = late.sort_values("DaysLate", ascending=False).head(5)
        st.markdown("**Top 5 Late Tasks (by days late)**")
        st.dataframe(top5[["Project", "TaskName", "Owner", "Finish", "DaysLate", "Critical", "PercentComplete", "Notes"]])

# Next 3 upcoming actions per project (open + future finish)
if not df_open_f.empty:
    upcoming = df_open_f.copy()
    upcoming["Finish"] = pd.to_datetime(upcoming["Finish"], errors="coerce")
    upcoming = upcoming[upcoming["Finish"].notna() & (upcoming["Finish"] >= today)]
    if not upcoming.empty:
        st.markdown("**Next 3 Upcoming Actions per Project**")
        rows = []
        for prj, grp in upcoming.groupby("Project"):
            g = grp.sort_values("Finish").head(3)
            rows.append(g)
        if rows:
            nxt = pd.concat(rows, ignore_index=True)
            show_cols = ["Project", "TaskName", "Finish", "Critical", "PercentComplete", "Notes"]
            st.dataframe(nxt[show_cols])

# ---------- Project Detail ----------
st.subheader("Project Detail")
if not df_all_tasks.empty:
    projs = [p for p in df_summary_f["ProjectName"].tolist() if p in df_all_tasks["Project"].unique().tolist()]
    if projs:
        tabs = st.tabs(projs)
        for tab, prj in zip(tabs, projs):
            with tab:
                tdf = df_all_tasks[df_all_tasks["Project"] == prj].copy()

                # Owners list for this project (EDIT #5)
                if not df_owner_pairs_all.empty:
                    owners_list = (
                        df_owner_pairs_all[df_owner_pairs_all["Project"] == prj]["Owner"].dropna().unique().tolist()
                    )
                else:
                    inline = (
                        tdf["ResourceNamesInline"].dropna().astype(str).str.replace(";", ",").str.split(",").explode()
                    )
                    owners_list = sorted({s.strip() for s in inline if s and s.strip()})

                st.markdown("**Project Resources**")
                if owners_list:
                    st.write(", ".join(sorted(set(owners_list))))
                else:
                    st.caption("No resources found for this project.")

                # EDIT #4: Bull/Bear block moved above Critical Path
                row = df_summary[df_summary["ProjectName"] == prj].iloc[0]
                st.markdown("**Bull / Bear Finish**")
                st.caption("Bull = planned finish; Bear = latest finish among open critical tasks (else open tasks/planned).")
                st.write(f"**Bull:** {row.get('BullFinish')}")
                st.write(f"**Bear:** {row.get('BearFinish')}")

                # Critical Path table (MSP-calculated 'Critical' flag)
                crit = tdf[(tdf["Critical"] == True) & (~tdf["IsSummary"])].copy()
                crit = crit.sort_values(["Finish", "Start", "TaskName"])
                st.markdown("**Critical Path (MS Project 'Critical' tasks):**")
                st.dataframe(crit[["TaskUID", "TaskName", "Start", "Finish", "PercentComplete"]])

                # Mini 3-across inside the tab (burndown + counts + placeholder heatmap per project)
                cta, ctb, ctc = st.columns(3)

                with cta:
                    odf = df_open_f[df_open_f["Project"] == prj].dropna(subset=["Finish"])
                    if not odf.empty:
                        bd = (
                            odf.assign(FinishDate=lambda d: pd.to_datetime(d["Finish"]).dt.date)
                            .groupby("FinishDate")
                            .size()
                            .reset_index(name="OpenTasks")
                            .sort_values("FinishDate")
                        )
                        small_line(bd, "FinishDate", "OpenTasks", "Burndown")

                with ctb:
                    open_n = int(df_open_f[df_open_f["Project"] == prj].shape[0]) if not df_open_f.empty else 0
                    crit_n = (
                        int((df_open_f[(df_open_f["Project"] == prj) & (df_open_f["Critical"] == True)]).shape[0])
                        if not df_open_f.empty
                        else 0
                    )
                    st.markdown("**Counts**")
                    st.write(f"Open actions: **{open_n}**")
                    st.write(f"Critical open: **{crit_n}**")

                with ctc:
                    # EDIT #6: Placeholder resource utilization heatmap (task counts per month)
                    # Build owner assignment table for this project
                    if not df_owner_pairs_all.empty:
                        owners_map = df_owner_pairs_all[df_owner_pairs_all["Project"] == prj][["TaskUID", "Owner"]]
                        tproj = tdf.merge(owners_map, on="TaskUID", how="left")
                    else:
                        tproj = tdf.copy()
                        tproj["Owner"] = (
                            tproj.get("ResourceNamesInline", "")
                            .astype(str).str.replace(";", ",").str.split(",").str[0].str.strip().fillna("Unassigned")
                        )

                    # Month from Start (fallback to Finish)
                    dates = pd.to_datetime(tproj["Start"]).fillna(pd.to_datetime(tproj["Finish"]))
                    tproj = tproj.assign(Month=dates.dt.to_period("M").astype(str))
                    util = (
                        tproj[(~tproj["IsSummary"]) & (tproj["TaskName"].str.len() > 0)]
                        .groupby(["Owner", "Month"]).size().reset_index(name="Tasks")
                    )
                    if not util.empty:
                        pivot = util.pivot(index="Owner", columns="Month", values="Tasks").fillna(0)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        im = ax.imshow(pivot.values, aspect="auto")
                        ax.set_title("Resource Util. (placeholder)")
                        ax.set_yticks(range(len(pivot.index)))
                        ax.set_yticklabels(pivot.index)
                        ax.set_xticks(range(len(pivot.columns)))
                        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                        # annotate
                        for i in range(pivot.shape[0]):
                            for j in range(pivot.shape[1]):
                                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=9)
                        fig.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.caption("No data for resource utilization heatmap (placeholder).")

# ---------- Portfolio-wide Resource Workload (EDIT #5) ----------
st.subheader("Portfolio Resource Workload (open actions)")
if not df_open_f.empty:
    if not df_owner_pairs_all.empty:
        owners_map_all = df_owner_pairs_all[["TaskUID", "Owner", "Project"]]
        tmp = df_open_f.merge(owners_map_all, on=["TaskUID", "Project"], how="left")
        tmp["Owner"] = tmp["Owner"].fillna("Unassigned")
    else:
        tmp = df_open_f.copy()
        tmp["Owner"] = tmp.get("ResourceNamesInline", "Unassigned").fillna("Unassigned")

    workload = (
        tmp.groupby(["Owner", "Project"])["TaskUID"].count().reset_index(name="OpenActions")
        .sort_values(["Owner", "OpenActions"], ascending=[True, False])
    )
    st.dataframe(workload)
