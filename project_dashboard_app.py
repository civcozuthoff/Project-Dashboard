import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# ================= App config & global plot style =================
st.set_page_config(page_title="Project Portfolio Dashboard", layout="wide")

# Smaller, cleaner typography (fits 2-across charts without overlap)
plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# ================= Constants =================
ns = {"msproj": "http://schemas.microsoft.com/project"}
# Store team notes in a CSV (point this to a shared path/UNC to collaborate)
# e.g., NOTES_STORE = r"\\CIVCO-FS\Projects\Dashboard\dashboard_notes.csv"
NOTES_STORE = "dashboard_notes.csv"

# ================= Utilities =================
def parse_dt(s: str | None):
    if not s:
        return None
    try:
        # MS Project XML can have trailing 'Z'
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

def load_notes():
    p = Path(NOTES_STORE)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame(columns=["Timestamp", "Project", "Author", "Email", "Note"])
    return pd.DataFrame(columns=["Timestamp", "Project", "Author", "Email", "Note"])

def append_note(project: str, author: str, email_to: str, note: str):
    if not note or not note.strip():
        return
    df = load_notes()
    new = pd.DataFrame([{
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Project": (project or "").strip(),
        "Author": (author or "Anonymous").strip(),
        "Email": (email_to or "").strip(),
        "Note": note.strip()
    }])
    out = pd.concat([df, new], ignore_index=True)
    out.to_csv(NOTES_STORE, index=False)

def send_email(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
               to_email: str, subject: str, body: str) -> tuple[bool, str]:
    """Send a simple plain-text email via SMTP. Returns (ok, message)."""
    try:
        if not (smtp_host and smtp_port and to_email):
            return False, "SMTP host/port or recipient email missing."
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user or "dashboard@localhost"
        msg["To"] = to_email
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            # Try STARTTLS if available
            try:
                server.starttls()
            except Exception:
                pass
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        return False, f"Email failed: {e}"

def small_line(df, xcol, ycol, title):
    fig, ax = plt.subplots(figsize=(6, 3.6))  # 2-across layout
    ax.plot(df[xcol], df[ycol], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.3)

    # Concise datetime ticks if applicable
    try:
        series = pd.Series(df[xcol])
        if pd.api.types.is_datetime64_any_dtype(series) or (len(series) and hasattr(series.iloc[0], "year")):
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    except Exception:
        pass

    fig.tight_layout()
    st.pyplot(fig)

def donut_chart(pct: float, title: str):
    """Draw a simple donut chart for a percentage."""
    pct = max(0.0, min(100.0, pct))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vals = [pct, 100 - pct]
    wedges, _ = ax.pie(vals, startangle=90)
    # donut hole
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.set_title(title)
    # center label
    ax.text(0, 0, f"{pct:.1f}%", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.axis("equal")
    st.pyplot(fig)

# ================= XML Parsing =================
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

    # ---- Resources ----
    res_uid_to_name = {}
    for r in root.findall("msproj:Resources/msproj:Resource", ns):
        uid = r.find("msproj:UID", ns)
        nm = r.find("msproj:Name", ns)
        if uid is not None and nm is not None and uid.text:
            res_uid_to_name[uid.text] = nm.text

    # ---- Tasks ----
    tasks = []
    for t in root.findall("msproj:Tasks/msproj:Task", ns):
        uid_el = t.find("msproj:UID", ns)
        if uid_el is None or uid_el.text == "0":
            continue

        def tg(tag, default=None):
            el = t.find(f"msproj:{tag}", ns)
            return el.text if el is not None and el.text is not None else default

        task_name = (tg("Name", "") or "").strip()
        if task_name == "":  # Exclude unnamed tasks
            continue

        summary = (tg("Summary", "0") == "1")
        percent = float(tg("PercentComplete", "0") or 0)
        active = (tg("Active", "1") == "1")
        start_task = parse_dt(tg("Start"))
        finish_task = parse_dt(tg("Finish"))
        actual_finish = parse_dt(tg("ActualFinish"))

        # --- Baseline (use first baseline if present) ---
        baseline_finish = None
        for bl in t.findall("msproj:Baseline", ns):
            fin_el = bl.find("msproj:Finish", ns)
            if fin_el is not None and fin_el.text:
                baseline_finish = parse_dt(fin_el.text)
                break  # take first available baseline finish

        critical = (tg("Critical", "0") == "1")
        milestone = (tg("Milestone", "0") == "1")
        notes = tg("Notes", "")
        res_inline = tg("ResourceNames", None)

        preds = []
        for pl in t.findall("msproj:PredecessorLink", ns):
            puid = pl.find("msproj:PredecessorUID", ns)
            if puid is not None and puid.text:
                preds.append(puid.text)

        # Strict "open": not summary, active, <100%, and no ActualFinish
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
            "BaselineFinish": baseline_finish,
            "Notes": notes,
            "ResourceNamesInline": res_inline,
            "Predecessors": preds,
            "IsOpenAction": is_open,
        })

    df_tasks = pd.DataFrame(tasks)

    # ---- Assignments → Owner mapping ----
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

    # Fallback to inline names if no assignments
    if df_pairs.empty and not df_tasks.empty:
        tmp = df_tasks[["TaskUID", "ResourceNamesInline"]].dropna()
        rows = []
        for _, r in tmp.iterrows():
            for part in str(r["ResourceNamesInline"]).replace(";", ",").split(","):
                nm = part.strip()
                if nm:
                    rows.append({"TaskUID": r["TaskUID"], "Owner": nm})
        df_pairs = pd.DataFrame(rows)

    # ---- Open/Late/Critical rollups ----
    non_summary = df_tasks[~df_tasks["IsSummary"]] if not df_tasks.empty else df_tasks
    open_actions = non_summary[non_summary["IsOpenAction"]] if not non_summary.empty else non_summary

    late_open = open_actions.copy()
    late_open = late_open[late_open["Finish"].notna() & (late_open["Finish"] < current_dt)]
    crit_open = open_actions[open_actions["Critical"] == True]

    overall_pct = round(non_summary["PercentComplete"].mean(), 1) if not non_summary.empty else 0.0

    # Bull/Bear
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

# ================= App Body =================
st.title("Project Portfolio Dashboard")

# ---- Data Source ----
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
            if tasks_df is not None and not task_frames: pass
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

# ---- Status classification ----
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

# ---- KPI Row ----
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Projects", len(df_summary))
with c2: st.metric("Total Open Actions", int(df_summary["OpenActions"].sum()))
with c3: st.metric("Critical Open Actions", int(df_summary["CriticalOpenActions"].sum()))
with c4: st.metric("Late Open Actions", int(df_summary["LateOpenActions"].sum()))

# ---- Status Table ----
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

# ---- Filters ----
st.sidebar.header("Filters")
status_filter = st.sidebar.multiselect(
    "Status", options=df_summary["Status"].unique().tolist(),
    default=df_summary["Status"].unique().tolist()
)
proj_filter = st.sidebar.multiselect(
    "Projects", options=df_summary["ProjectName"].tolist(),
    default=df_summary["ProjectName"].tolist()
)

df_summary_f = df_summary[df_summary["Status"].isin(status_filter) & df_summary["ProjectName"].isin(proj_filter)]

if df_all_open is not None and not df_all_open.empty:
    df_open_f = df_all_open[df_all_open["Project"].isin(df_summary_f["ProjectName"])].copy()
else:
    df_open_f = pd.DataFrame()

# ================= Charts (2 across) =================
colA, colB = st.columns(2)

# Portfolio Burndown (month buckets for readability)
if not df_open_f.empty:
    bd_overall = (
        df_open_f[~df_open_f["IsSummary"]]
        .dropna(subset=["Finish"])
        .assign(FinishDate=lambda d: pd.to_datetime(d["Finish"]).dt.to_period("M").dt.to_timestamp())
        .groupby("FinishDate").size().reset_index(name="OpenTasks")
        .sort_values("FinishDate")
    )
    with colA:
        if not bd_overall.empty:
            small_line(bd_overall, "FinishDate", "OpenTasks", "Burndown — Portfolio (filtered)")

# Owner Load (Assignments preferred)
if not df_open_f.empty:
    if not df_owner_pairs_all.empty:
        owners_map = df_owner_pairs_all[df_owner_pairs_all["Project"].isin(df_summary_f["ProjectName"])][["TaskUID", "Owner"]]
        tmp = df_open_f.merge(owners_map, on="TaskUID", how="left")
        tmp["Owner"] = tmp["Owner"].fillna("Unassigned")
    else:
        tmp = df_open_f.copy()
        tmp["Owner"] = tmp.get("ResourceNamesInline", "Unassigned").fillna("Unassigned")
    owner_counts = tmp[~tmp["IsSummary"]].groupby("Owner")["TaskUID"].count().sort_values(ascending=False).reset_index(name="OpenActions")
    with colB:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(owner_counts["Owner"], owner_counts["OpenActions"])
        ax.set_title("Open Actions by Owner")
        ax.set_xlabel("Owner"); ax.set_ylabel("OpenActions")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

# Next row
colC, colD = st.columns(2)

# Aging buckets
if not df_open_f.empty:
    df_open_f["Finish"] = pd.to_datetime(df_open_f["Finish"], errors="coerce")
    df_open_f["DaysLate"] = (today - df_open_f["Finish"]).dt.days

    def bucket(d):
        if pd.isna(d) or d < 0: return "Not Due"
        if d <= 7: return "0–7"
        if d <= 14: return "8–14"
        return ">14"

    df_open_f["AgingBucket"] = df_open_f["DaysLate"].apply(bucket)
    age_tbl = (
        df_open_f[(df_open_f["AgingBucket"] != "Not Due") & (~df_open_f["IsSummary"])]
        .groupby(["Project", "AgingBucket"]).size().reset_index(name="Tasks")
        .pivot(index="Project", columns="AgingBucket", values="Tasks").fillna(0)
        .reindex(columns=["0–7", "8–14", ">14"], fill_value=0)
    )
    with colC:
        if not age_tbl.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3.6))
            bottom = np.zeros(len(age_tbl))
            for col in age_tbl.columns:
                vals = age_tbl[col].values
                ax2.bar(age_tbl.index, vals, bottom=bottom, label=col)
                bottom = bottom + vals
            ax2.set_title("Aging by Project")
            ax2.set_xlabel("Project"); ax2.set_ylabel("Tasks")
            ax2.legend(title="Days Late")
            ax2.tick_params(axis="x", rotation=45)
            fig2.tight_layout()
            st.pyplot(fig2)

with colD:
    st.empty()

# ================= Master Resource Utilization (Portfolio) =================
st.subheader("Master Resource Utilization (Open Tasks per Month)")
if not df_all_tasks.empty:
    # Build Owner mapping across all tasks
    if not df_owner_pairs_all.empty:
        owners_map_all = df_owner_pairs_all[["TaskUID", "Owner", "Project"]]
        t_all = df_all_tasks.merge(owners_map_all, on=["TaskUID", "Project"], how="left")
        t_all["Owner"] = t_all["Owner"].fillna("Unassigned")
    else:
        t_all = df_all_tasks.copy()
        t_all["Owner"] = (
            t_all.get("ResourceNamesInline", "")
            .astype(str).str.replace(";", ",").str.split(",").str[0].str.strip().fillna("Unassigned")
        )
    # Use OPEN tasks only for utilization (current workload proxy)
    open_all = t_all[t_all["IsOpenAction"] & (~t_all["IsSummary"]) & (t_all["TaskName"].str.len() > 0)].copy()
    if not open_all.empty:
        dates = pd.to_datetime(open_all["Start"]).fillna(pd.to_datetime(open_all["Finish"]))
        open_all = open_all.assign(Month=dates.dt.to_period("M").astype(str))
        util = open_all.groupby(["Owner", "Month"]).size().reset_index(name="Tasks")
        pivot = util.pivot(index="Owner", columns="Month", values="Tasks").fillna(0)
        figU, axU = plt.subplots(figsize=(10, 6))
        im = axU.imshow(pivot.values, aspect="auto")
        axU.set_title("Open Task Count per Resource per Month (proxy for hours)")
        axU.set_yticks(range(len(pivot.index))); axU.set_yticklabels(pivot.index)
        axU.set_xticks(range(len(pivot.columns))); axU.set_xticklabels(pivot.columns, rotation=45, ha="right")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                axU.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=8)
        figU.tight_layout()
        st.pyplot(figU)
    else:
        st.caption("No open tasks available to build utilization.")

# ================= Resource Explorer =================
st.subheader("Resource Explorer (Active Tasks)")
if not df_all_tasks.empty:
    # Build list of owners from assignments or inline
    if not df_owner_pairs_all.empty:
        owners_list = sorted(df_owner_pairs_all["Owner"].dropna().unique().tolist())
    else:
        inline = (
            df_all_tasks["ResourceNamesInline"].dropna()
            .astype(str).str.replace(";", ",").str.split(",").explode()
        )
        owners_list = sorted({s.strip() for s in inline if s and s.strip()})
    if owners_list:
        selected_owner = st.selectbox("Select a resource", owners_list)
        if selected_owner:
            # Map owners to tasks (best-effort)
            if not df_owner_pairs_all.empty:
                m = df_owner_pairs_all[df_owner_pairs_all["Owner"] == selected_owner][["TaskUID", "Project"]]
                my_tasks = df_all_tasks.merge(m, on=["TaskUID", "Project"], how="inner")
            else:
                my_tasks = df_all_tasks[df_all_tasks["ResourceNamesInline"].fillna("").str.contains(selected_owner, case=False, na=False)].copy()
            # Filter active, non-summary, named
            my_open = my_tasks[my_tasks["IsOpenAction"] & (~my_tasks["IsSummary"]) & (my_tasks["TaskName"].str.len() > 0)]
            st.markdown(f"**Active Tasks for:** {selected_owner}")
            if not my_open.empty:
                st.dataframe(
                    my_open[["Project","TaskName","Start","Finish","Critical","PercentComplete","Notes"]]
                    .sort_values(["Project","Finish","TaskName"])
                )
                # quick workload by project
                w = my_open.groupby("Project")["TaskUID"].count().reset_index(name="OpenActions").sort_values("OpenActions", ascending=False)
                figW, axW = plt.subplots(figsize=(6, 3.6))
                axW.bar(w["Project"], w["OpenActions"])
                axW.set_title("Open Actions by Project")
                axW.set_xlabel("Project"); axW.set_ylabel("OpenActions")
                axW.tick_params(axis="x", rotation=45)
                figW.tight_layout()
                st.pyplot(figW)
            else:
                st.caption("No active tasks for this resource.")
    else:
        st.caption("No resources found.")

# ================= Focus: Late & Upcoming =================
st.subheader("Focus: Late and Upcoming")

# Top 5 Late (strict open, not summary, named)
if not df_open_f.empty:
    late = df_open_f[~df_open_f["IsSummary"]].copy()
    late["Finish"] = pd.to_datetime(late["Finish"], errors="coerce")
    late = late[
        (late["TaskName"].str.len() > 0) &
        late["Finish"].notna() &
        (late["Finish"] < today) &
        (late["PercentComplete"] < 100) &
        (late["ActualFinish"].isna())
    ]
    if not late.empty:
        late["DaysLate"] = (today - late["Finish"]).dt.days
        if "Owner" not in late.columns and not df_owner_pairs_all.empty:
            owners_map_all = df_owner_pairs_all[["TaskUID", "Owner"]]
            late = late.merge(owners_map_all, on="TaskUID", how="left")
        late["Owner"] = late.get("Owner", late.get("ResourceNamesInline", "Unassigned")).fillna("Unassigned")
        top5 = late.sort_values("DaysLate", ascending=False).head(5)
        st.markdown("**Top 5 Late Tasks (by days late)**")
        st.dataframe(top5[["Project","TaskName","Owner","Finish","DaysLate","Critical","PercentComplete","Notes"]])

# Next 3 Upcoming (strict open, not summary, named)
if not df_open_f.empty:
    upcoming = df_open_f[~df_open_f["IsSummary"]].copy()
    upcoming["Finish"] = pd.to_datetime(upcoming["Finish"], errors="coerce")
    upcoming = upcoming[
        (upcoming["TaskName"].str.len() > 0) &
        (upcoming["Finish"].notna()) &
        (upcoming["Finish"] >= today) &
        (upcoming["PercentComplete"] < 100) &
        (upcoming["ActualFinish"].isna())
    ]
    if not upcoming.empty:
        st.markdown("**Next 3 Upcoming Actions per Project**")
        rows = []
        for prj, grp in upcoming.groupby("Project"):
            rows.append(grp.sort_values("Finish").head(3))
        nxt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not nxt.empty:
            show_cols = ["Project","TaskName","Finish","Critical","PercentComplete","Notes"]
            st.dataframe(nxt[show_cols])

# ================= On-Time Task Completion =================
st.subheader("On-Time Task Completion %")

# Completed, non-summary tasks with planned (BaselineFinish or Finish) + ActualFinish
if not df_all_tasks.empty:
    dfc = df_all_tasks.copy()
    dfc = dfc[
        (~dfc["IsSummary"]) &
        (dfc["ActualFinish"].notna()) &
        (dfc["Finish"].notna())
    ].copy()
    dfc["PlannedFinish"] = dfc["BaselineFinish"].where(dfc["BaselineFinish"].notna(), dfc["Finish"])
    dfc["OnTime"] = pd.to_datetime(dfc["ActualFinish"]) <= pd.to_datetime(dfc["PlannedFinish"])
    dfc = dfc[dfc["Project"].isin(df_summary_f["ProjectName"])]

    if not dfc.empty:
        overall_pct = float(round(100 * dfc["OnTime"].mean(), 1))
        oa, ob = st.columns(2)
        with oa:
            donut_chart(overall_pct, "Overall On-Time % (completed tasks)")
        with ob:
            proj_rates = (
                dfc.groupby("Project")["OnTime"]
                   .mean().mul(100).round(1)
                   .reset_index(name="OnTimePct")
                   .sort_values("OnTimePct", ascending=False)
            )
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            ax3.bar(proj_rates["Project"], proj_rates["OnTimePct"])
            ax3.set_title("On-Time % by Project")
            ax3.set_xlabel("Project"); ax3.set_ylabel("On-Time %")
            ax3.set_ylim(0, 100)
            ax3.tick_params(axis="x", rotation=45)
            fig3.tight_layout()
            st.pyplot(fig3)

        # Monthly trend by ActualFinish month
        st.markdown("**Monthly Trend – On-Time Completion**")
        dfc["Month"] = pd.to_datetime(dfc["ActualFinish"]).dt.to_period("M").astype(str)
        trend = (
            dfc.groupby("Month")["OnTime"]
               .mean().mul(100).round(1)
               .reset_index(name="OnTimePct")
               .sort_values("Month")
        )
        if not trend.empty:
            fig4, ax4 = plt.subplots(figsize=(14, 8))
            ax4.plot(trend["Month"], trend["OnTimePct"], marker="o")
            ax4.set_title("On-Time Completion % by Month")
            ax4.set_xlabel("Month"); ax4.set_ylabel("On-Time %")
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
            fig4.tight_layout()
            st.pyplot(fig4)
        else:
            st.caption("No on-time completion trend data available.")
    else:
        st.caption("No completed tasks with both Actual Finish and planned dates found.")
else:
    st.caption("No task data available yet.")

# ================= Project Notes (shared + optional email) =================
st.subheader("Project Notes (shared)")
st.caption(f"Notes are saved to: **{NOTES_STORE}** (point this to a shared path/SharePoint sync for team visibility).")

note_col1, note_col2 = st.columns([2, 3])
with note_col1:
    prj_for_note = st.selectbox("Select project", df_summary_f["ProjectName"].tolist())
    author = st.text_input("Your name", value="")
    owner_email = st.text_input("Recipient email (e.g., project owner)", value="", placeholder="owner@company.com")
    note_text = st.text_area("Add a note (stored in CSV, can email if configured)", height=100, placeholder="Type an update or decision…")
    email_toggle = st.checkbox("Email this note to recipient", value=False)

    # Simple SMTP settings (optional)
    with st.expander("Email settings (SMTP)", expanded=False):
        smtp_host = st.text_input("SMTP server (host)", value="")
        smtp_port = st.number_input("SMTP port", value=587, step=1)
        smtp_user = st.text_input("SMTP username (optional)", value="")
        smtp_pass = st.text_input("SMTP password (optional)", type="password", value="")

    if st.button("Save note"):
        append_note(prj_for_note, author, owner_email, note_text)
        st.success("Note saved.")
        if email_toggle and owner_email:
            ok, msg = send_email(
                smtp_host=smtp_host, smtp_port=int(smtp_port),
                smtp_user=smtp_user, smtp_pass=smtp_pass,
                to_email=owner_email,
                subject=f"[Project {prj_for_note}] Dashboard Note from {author or 'Anonymous'}",
                body=note_text
            )
            if ok:
                st.success("Email sent to recipient.")
            else:
                st.warning(msg)

with note_col2:
    notes_df = load_notes()
    if not notes_df.empty:
        show_all = st.checkbox("Show all project notes", value=False)
        if not show_all:
            notes_df = notes_df[notes_df["Project"] == prj_for_note]
        st.dataframe(notes_df.sort_values("Timestamp", ascending=False))
    else:
        st.caption("No notes yet. Be the first to add one!")

# ================= Master Project Timeline (Bull vs Bear) =================
st.subheader("Master Timeline (Bull vs Bear Finish)")
if not df_summary_f.empty:
    # Build data
    tline = df_summary_f[["ProjectName","BullFinish","BearFinish"]].copy()
    tline["BullFinish"] = pd.to_datetime(tline["BullFinish"])
    tline["BearFinish"] = pd.to_datetime(tline["BearFinish"])
    tline = tline.dropna(subset=["BullFinish", "BearFinish"])
    if not tline.empty:
        tline = tline.sort_values("BearFinish")  # sort by Bear (latest)
        y = np.arange(len(tline))
        figT, axT = plt.subplots(figsize=(10, 0.5 * len(tline) + 1.5))
        for i, row in enumerate(tline.itertuples(index=False)):
            lo = min(row.BullFinish, row.BearFinish)
            hi = max(row.BullFinish, row.BearFinish)
            axT.hlines(y=i, xmin=lo, xmax=hi, color="tab:blue", linewidth=3, alpha=0.6)
            axT.plot(lo, i, "o", label="Bull" if i == 0 else "", markersize=6)
            axT.plot(hi, i, "s", label="Bear" if i == 0 else "", markersize=6)
        axT.set_yticks(y)
        axT.set_yticklabels(tline["ProjectName"])
        axT.set_title("Bull (●) and Bear (■) Completion Dates by Project")
        axT.grid(axis="x", alpha=0.2)
        locator = AutoDateLocator()
        axT.xaxis.set_major_locator(locator)
        axT.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        if len(tline) > 1:
            axT.legend(loc="upper right")
        figT.tight_layout()
        st.pyplot(figT)
    else:
        st.caption("No Bull/Bear finish dates available to plot.")

# ================= Project Detail =================
st.subheader("Project Detail")
if not df_all_tasks.empty:
    projs = [p for p in df_summary_f["ProjectName"].tolist()
             if p in df_all_tasks["Project"].unique().tolist()]
    if projs:
        tabs = st.tabs(projs)
        for tab, prj in zip(tabs, projs):
            with tab:
                tdf = df_all_tasks[df_all_tasks["Project"] == prj].copy()

                # Project Resources list
                if not df_owner_pairs_all.empty:
                    owners_list = (
                        df_owner_pairs_all[df_owner_pairs_all["Project"] == prj]["Owner"]
                        .dropna().unique().tolist()
                    )
                else:
                    inline = (
                        tdf["ResourceNamesInline"].dropna()
                        .astype(str).str.replace(";", ",").str.split(",").explode()
                    )
                    owners_list = sorted({s.strip() for s in inline if s and s.strip()})

                st.markdown("**Project Resources**")
                if owners_list:
                    st.write(", ".join(sorted(set(owners_list))))
                else:
                    st.caption("No resources found for this project.")

                # Bull/Bear FIRST (above Critical Path)
                row = df_summary[df_summary["ProjectName"] == prj].iloc[0]
                st.markdown("**Bull / Bear Finish**")
                st.caption("Bull = planned finish; Bear = latest finish among open critical tasks (else open tasks/planned).")
                st.write(f"**Bull:** {row.get('BullFinish')}")
                st.write(f"**Bear:** {row.get('BearFinish')}")

                # Critical Path (as flagged by MS Project)
                crit = tdf[(tdf["Critical"] == True) & (~tdf["IsSummary"])].copy()
                crit = crit.sort_values(["Finish", "Start", "TaskName"])
                st.markdown("**Critical Path (MS Project 'Critical' tasks):**")
                st.dataframe(crit[["TaskUID","TaskName","Start","Finish","PercentComplete"]])

                # Mini 2-across inside the tab: burndown + counts + placeholder heatmap
                cta, ctb = st.columns(2)

                with cta:
                    odf = df_all_open[(df_all_open["Project"] == prj)]
                    odf = odf.dropna(subset=["Finish"])
                    if not odf.empty:
                        bd = (
                            odf.assign(FinishDate=lambda d: pd.to_datetime(d["Finish"]).dt.to_period("M").dt.to_timestamp())
                            .groupby("FinishDate").size().reset_index(name="OpenTasks")
                            .sort_values("FinishDate")
                        )
                        small_line(bd, "FinishDate", "OpenTasks", "Burndown")

                with ctb:
                    open_n = int(df_all_open[df_all_open["Project"] == prj].shape[0]) if not df_all_open.empty else 0
                    crit_n = int(
                        df_all_open[(df_all_open["Project"] == prj) & (df_all_open["Critical"] == True)].shape[0]
                    ) if not df_all_open.empty else 0
                    st.markdown("**Counts**")
                    st.write(f"Open actions: **{open_n}**")
                    st.write(f"Critical open: **{crit_n}**")

                # Placeholder Resource Utilization heatmap (task counts per month)
                if not df_owner_pairs_all.empty:
                    owners_map = df_owner_pairs_all[df_owner_pairs_all["Project"] == prj][["TaskUID", "Owner"]]
                    tproj = tdf.merge(owners_map, on=["TaskUID"], how="left")
                else:
                    tproj = tdf.copy()
                    tproj["Owner"] = (
                        tproj.get("ResourceNamesInline", "")
                        .astype(str).str.replace(";", ",").str.split(",").str[0].str.strip().fillna("Unassigned")
                    )

                dates = pd.to_datetime(tproj["Start"]).fillna(pd.to_datetime(tproj["Finish"]))
                tproj = tproj.assign(Month=dates.dt.to_period("M").astype(str))
                util = (
                    tproj[(~tproj["IsSummary"]) & (tproj["TaskName"].str.len() > 0)]
                    .groupby(["Owner", "Month"]).size().reset_index(name="Tasks")
                )
                st.markdown("**Resource Utilization (placeholder)**")
                if not util.empty:
                    pivot = util.pivot(index="Owner", columns="Month", values="Tasks").fillna(0)
                    fig5, ax5 = plt.subplots(figsize=(6, 3.6))
                    ax5.imshow(pivot.values, aspect="auto")
                    ax5.set_title("Task Count per Month (proxy for hours)")
                    ax5.set_yticks(range(len(pivot.index))); ax5.set_yticklabels(pivot.index)
                    ax5.set_xticks(range(len(pivot.columns))); ax5.set_xticklabels(pivot.columns, rotation=45, ha="right")
                    for i in range(pivot.shape[0]):
                        for j in range(pivot.shape[1]):
                            ax5.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=8)
                    fig5.tight_layout()
                    st.pyplot(fig5)
                else:
                    st.caption("No data for utilization heatmap (placeholder).")
