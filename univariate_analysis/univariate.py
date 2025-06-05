import math
from itertools import combinations_with_replacement, product

import anndata as ad
import forestplot as fp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyComplexHeatmap as pch
import scanpy as sc
import seaborn as sns
import statsmodels.api as sm
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from napari_prism.models.adata_ops.feature_modelling._discrete import (
    _consolidate_statistics,
    _normalise_log2p,
    difference_of_means_by_binary_label,
    univariate_test_feature_by_binary_label,
)

# Get adata
from napari_prism.models.adata_ops.feature_modelling._obs import ObsAggregator
from napari_prism.models.adata_ops.feature_modelling._survival import (
    get_sample_level_adata,
    kaplan_meier,
    parse_survival_columns,
    plot_kaplan_meier,
)
from numpy.linalg import LinAlgError
from pandas.api.types import CategoricalDtype
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


class AttrDict(dict):
    __getattr__ = dict.__getitem__


# Define global vars
global label_encodings
global color_encodings
cmaps = AttrDict()
color_encodings = AttrDict()
label_encodings = AttrDict()
input_adata_path = "3-output_nsclc_ytma_features_preandpost_withMetabolicFeatures.h5ad"
input_clinical_sheet = ""  # Input the xlsx file her
input_clinical_sheet_471 = ""  # Input the sheet tab name of the 471 cohort
input_clinical_sheet_404 = ""  # Input the sheet tab name of the 404 cohort

immunefunc_markers = [
    "pd1",
    "pdl1",
    "pdl1high",
    "granzymeb",
    "icos",
    "ido1",
    "ido1high",
]
immunemeta_markers = [
    "asct2",
    "asct2high",
    "atpa5",
    "atpa5high",
    "citratesynthase",
    "citratesynthasehigh",
    "cpt1a",
    "cpt1ahigh",
    "g6pd",
    "g6pdhigh",
    "glut1",
    "glut1high",
    "hexokinase1",
    "hexokinase1high",
    "idh2",
    "idh2high",
    "nakatpase",
    "pnrf2",
    "pnrf2high",
    "sdha",
    "sdhahigh",
]
tumorfunc_markers = [
    "pdl1",
    "pdl1high",
    "vimentin",
    "ki67",
    "ido1",
    "ido1high",
    "hlaa",
    "hlaahigh",
]
tumormeta_markers = [
    "asct2",
    "asct2high",
    "atpa5",
    "atpa5high",
    "citratesynthase",
    "citratesynthasehigh",
    "cpt1a",
    "cpt1ahigh",
    "g6pd",
    "g6pdhigh",
    "glut1",
    "glut1high",
    "hexokinase1",
    "hexokinase1high",
    "idh2",
    "idh2high",
    "nakatpase",
    "pnrf2",
    "pnrf2high",
    "sdha",
    "sdhahigh",
]


def add_save_path_suffix(save_path, suffix, delimiter="_"):
    def _add_suffix(save_path, suffix, delimiter):
        assert "." in save_path
        fname = save_path.split(".")[0]
        extension = save_path.split(".")[1]
        return f"{fname}{delimiter}{suffix}.{extension}"

    if save_path is None:
        return None
    else:
        if isinstance(save_path, list):
            return [_add_suffix(s, suffix, delimiter) for s in save_path]
        else:
            return _add_suffix(save_path, suffix, delimiter)


def create_cmaps():
    custom = LinearSegmentedColormap.from_list(
        "my_gradient",
        (
            # Edit this gradient at https://eltos.github.io/gradient/#C000EE-EAEAEA-038F00
            (0.000, (0.753, 0.000, 0.933)),
            (0.500, (0.918, 0.918, 0.918)),
            (1.000, (0.012, 0.561, 0.000)),
        ),
    )
    cmaps["dsmash"] = custom


def create_meta_path_map():
    meta_path_index_to_str = {
        3: "Minimal_Metabolic_Activity",
        2: "Medium_And_Regulatory_Activity",
        0: "Low_Metabolic_Activity",
        1: "High_Metabolic_Activity",
    }
    label_encodings["meta_path"] = meta_path_index_to_str


def create_cell_type_colors():
    cell_types_colors = {
        "tumorcells": "#d6336f",
        "bcells": "#4a65ea",  # Dark blue
        "plasmacells": "#4aa0ea",  # a lighter blue; related to b cells
        "cd4tcells": "#49ad37",  # base green
        "cd4tregcells": "#c4d33b",  # yellowey green. Less toxic function, more 'helping', T cell logistics
        "cd8tcells": "#22d595",  # a more toxic, brighter turq,green to indicate killing function
        "endothilialcells": "#da6fd2",  # pink, related to blood. avoid red to not confused with 'detrimental' cells like tumor cells
        "fibroblastcells": "#622d04",  # brown, reminiscent of VIM staining with HRP DAB
        "myofibroblastcells": "#d86e53",  # a redder brown / orange
        # "myeloid_cells": "#8f56d5", # Base Purple for Myeloids
        "granulocytecells": "#4a118a",  # Darker purple to represent more density and size
        "macrophagecells": "#9023f6",  # Brighter purple
        # Other cell types, base
        "immunenoscells": "#87b09a",  # muted green
        "myeloidnoscells": "#b292d8",  # Muted version of hte myeloid_cells
        "othercells": "#86807a",  # Muted version of the other cells
        # "artifactcells": "#cfcec1", # Muted version of the other cells
    }
    color_encodings["cell_types"] = cell_types_colors


def create_metadata_colors():
    # metadata colors defined here; can store into globals
    or_colors = {
        "Responder": "#365ba2",
        "Non-responder": "#a0464b",
    }
    color_encodings["OR"] = or_colors

    bor_colors = {
        "NE": "#a19b9c",
        "CR": "#365ba2",
        "PR": "#7159a6",
        "SD": "#a0239d",
        "PD": "#a0464b",
    }
    color_encodings["BOR"] = bor_colors

    cb6_colors = {"Yes": "#365ba2", "No": "#a0464b"}
    color_encodings["CB6"] = cb6_colors

    gender_colors = {
        "Female": "#FD6320",  # Orange
        "Male": "#58897D",  # Green
    }
    color_encodings["Gender"] = gender_colors

    smoking_colors = {
        "Current": "#434f4b",  # Dark Grey, tint green
        "Former": "#869892",  # Lighter Grey, lgohter green
        "Never": "#d0d0d0",  # even lighter Grey, no tint
    }
    color_encodings["Smoking_Status"] = smoking_colors

    histology_colors = {
        "Adenocarcinoma": "#26044f",  # better sruvival than squamous;
        "Squamous": "#0f7d38",  # olivey green; squamous only?
        "Adenosquamous": "#a08622",  # green; squamous + glandlike cells
    }
    color_encodings["Histology"] = histology_colors

    stage_colors = {
        "I": "#f4d8cd",  # light peach
        "II": "#e2bdc8",  # peach
        "III": "#cb5f81",  #
        "IV": "#8f002c",  # red
    }
    color_encodings["Stage"] = stage_colors

    biopsy_colors = {
        "Small bowel": "#9b470b",  # brown
        "Brain": "#ce7cbc",  # pink
        "Spinal": "#9b988f",  # muted sand, grey
        "Lymph Node": "#a5aa5e",  # muted yellow
        "Lung": "#80b18d",  # muted green
        "Skin": "#c9b47b",  # muted sand
    }
    color_encodings["Biopsy"] = biopsy_colors

    pp_colors = {"Pre": "#8b94a5", "Post": "#474b52"}
    color_encodings["PrePost"] = pp_colors


def load_process_input_adata(adata_path):
    adata = ad.read_h5ad(adata_path)

    # Rename columns with underscore for better readability
    adata.obs = adata.obs.rename(
        columns={
            "celltypes": "cell_types",
            "newcelltypes": "new_cell_types",
            "patientid": "patient_id",
            "biopsysite": "biopsy_site",
            "histologytype": "histology_type",
        }
    )

    adata.obs["patient_id"] = adata.obs["patient_id"].astype(str)
    original_obs = adata.obs.columns
    return adata, original_obs


def extend_clinical_metadata(adata, sheet, sheet_471, sheet_404):
    clinical_471 = pd.read_excel(sheet, sheet_name=sheet_471)
    clinical_404 = pd.read_excel(sheet, sheet_name=sheet_404)
    clinical_nsclc = pd.concat([clinical_471, clinical_404])
    clinical_nsclc["CPID"] = clinical_nsclc["CPID"].astype(str)
    clinical_nsclc["SPID"] = clinical_nsclc["SPID"].astype(str)

    obs_extended = pd.merge(
        left=adata.obs,
        right=clinical_nsclc,
        left_on="patient_id",
        right_on="CPID",
        suffixes=("", "_y"),
        how="left",
    )

    obs_extended = obs_extended.drop(
        obs_extended.filter(regex="_y$").columns,
        axis=1,
    )
    clinical_nsclc[clinical_nsclc["CPID"] == "22193"]

    obs_extended.index = obs_extended.index.astype(str)

    # Drop duplicated records
    obs_extended = obs_extended[
        ~((obs_extended.CPID == "22193") & (obs_extended.SPID.isin(["35119", "22519"])))
    ]

    assert obs_extended.shape[0] == adata.obs.shape[0]

    adata.obs = obs_extended

    # merge cols.
    adata.obs["Age"] = adata.obs["Age"].fillna(adata.obs["Age_at_first_dose_of_IT"])
    # Smoking status in 471 is "Smoking_Status", in 404 is "Tobacco_History"
    adata.obs["Smoking_Status"] = adata.obs["Smoking_Status"].fillna(
        adata.obs["Tobacco_History"]
    )
    adata.obs["Stage_at_ITx"] = adata.obs["Stage_at_ITx"].fillna(
        adata.obs["Stage_at_the_beginning_of_IT_treatment"]
    )

    # Sanitise
    gender_map = {"Female ": "Female", "Female": "Female", "Male": "Male"}
    adata.obs["Gender"] = adata.obs["Gender"].replace(gender_map)

    stage_map = {
        "IB": "I",
        "Stage IV (M1c)": "IV",
        "Stage IV (M1b)": "IV",
        "Stage IV (M1a)": "IV",
        "Stage IV (M1)": "IV",
        "Extensive": "IV",  # Assume two-stage extensive as IV
        "IIB": "II",
        "IIIA": "III",
        "IIIB": "III",
    }
    adata.obs["Stage_at_ITx"] = adata.obs["Stage_at_ITx"].replace(stage_map)

    biopsy_site_map = {
        "Bowel": "Small bowel",
        "Brain": "Brain",
        "Lung": "Lung",
        "Lung primary": "Lung",
        "Metastatic Soft tissue from spine (with bone?)": "Spinal",
        "Mediastinal lymph node": "Lymph Node",
        "LN": "Lymph Node",
        "Brain metastasis": "Brain",
        "Neck lymph node": "Lymph Node",
        "Spinal": "Spinal",
        "Axillary lymph node": "Lymph Node",
        "Skin metastasis": "Skin",
    }
    adata.obs["biopsy_site"] = adata.obs["biopsy_site"].replace(biopsy_site_map)

    # Consistent smoking categories;
    smoking_map = {
        "Current smoker": "Current",
        "Never smoker": "Never",
        "Former smoker": "Former",
    }
    adata.obs["Smoking_Status"] = adata.obs["Smoking_Status"].replace(smoking_map)
    responder_map = {0.0: "Non-responder", 1.0: "Responder"}
    adata.obs["OR"] = adata.obs["OR"].replace(responder_map)

    cb6_map = {0.0: "No", 1.0: "Yes"}
    adata.obs["CB6"] = adata.obs["CB6"].replace(cb6_map)

    ordinal_bor = ["NE", "CR", "PR", "SD", "PD"]

    ordinal_bor = ordinal_bor[::-1]

    ordinal_bor_dtype = CategoricalDtype(categories=ordinal_bor, ordered=True)

    adata.obs["BOR"] = adata.obs["BOR"].astype(ordinal_bor_dtype)
    print(f"\t {adata.shape[0]} cells, {adata.obs['patient_id'].nunique()} patients")
    return adata


def figure_1c_heatmap(
    adata,
    figsize=(14, 10),
    dpi=500,
    save_path=None,
):
    agg = ObsAggregator(adata, "patient_id")

    # Ys
    bor = agg.get_metadata_df("BOR")
    ovr = agg.get_metadata_df("OR")
    cb6 = agg.get_metadata_df("CB6")

    age = agg.get_metadata_df("Age")
    gender = agg.get_metadata_df("Gender")
    smoking = agg.get_metadata_df("Smoking_Status")
    histology = agg.get_metadata_df("histology_type")
    stage = agg.get_metadata_df("Stage_at_ITx")
    biopsy = agg.get_metadata_df("biopsy_site")

    pp = agg.get_metadata_df("prepost")

    # Time/Continuous
    pfs = agg.get_metadata_df("pfs.T")
    os = agg.get_metadata_df("os.T")

    # Custom sorting;
    biopsy_order = (
        biopsy["biopsy_site"].value_counts().sort_values(ascending=False).index.tolist()
    )
    biopsy["biopsy_site"] = pd.Categorical(
        biopsy["biopsy_site"], categories=biopsy_order, ordered=True
    )
    biopsy = biopsy.sort_values("biopsy_site")

    all_metadata = pd.concat(
        [bor, ovr, cb6, age, gender, smoking, histology, stage, biopsy, pfs, os, pp],
        axis=1,
    )
    all_metadata = all_metadata.sort_values(
        [
            "CB6",
            "OR",
            "BOR",
            "biopsy_site",
            "Stage_at_ITx",
            "histology_type",
            "Smoking_Status",
        ],
        ascending=True,
    )

    merged_time = pd.merge(
        agg.get_metadata_df("os.T"),
        agg.get_metadata_df("pfs.T"),
        left_index=True,
        right_index=True,
    )
    merged_time["os.T"] = merged_time["os.T"] - merged_time["pfs.T"]
    merged_time = merged_time.loc[all_metadata.index]
    hw = 3
    lw = 3
    ec = None
    row_kwargs = {
        "Age": pch.anno_scatterplot(all_metadata["Age"], height=4, colors="k"),
        "Gender": pch.anno_simple(
            all_metadata["Gender"],
            colors=color_encodings["Gender"],
            height=0.75,
            linewidths=lw,
            edgecolors=ec,
        ),
        "Smoking": pch.anno_simple(
            all_metadata["Smoking_Status"],
            colors=color_encodings["Smoking_Status"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "Histology": pch.anno_simple(
            all_metadata["histology_type"],
            colors=color_encodings["Histology"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "Stage": pch.anno_simple(
            all_metadata["Stage_at_ITx"],
            colors=color_encodings["Stage"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "Biopsy": pch.anno_simple(
            all_metadata["biopsy_site"],
            colors=color_encodings["Biopsy"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "Pre/Post": pch.anno_simple(
            all_metadata["prepost"],
            colors=color_encodings["PrePost"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "BOR": pch.anno_simple(
            all_metadata["BOR"],
            colors=color_encodings["BOR"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "OR": pch.anno_simple(
            all_metadata["OR"],
            colors=color_encodings["OR"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "CB6": pch.anno_simple(
            all_metadata["CB6"],
            colors=color_encodings["CB6"],
            height=hw,
            linewidths=lw,
            edgecolors=ec,
        ),
        "PFS/OS": pch.anno_barplot(
            merged_time[["pfs.T", "os.T"]],
            colors={"pfs.T": "dodgerblue", "os.T": "grey"},
            height=20,
            width=0.6,
            legend=True,
        ),
    }
    plt.figure(figsize=figsize, dpi=dpi)
    metadata_annot = pch.HeatmapAnnotation(
        axis=1, wgap=1, hgap=1, legend=False, **row_kwargs
    )
    metadata_annot.plot_annotations()
    all_ax = plt.gcf().axes
    time_ax = [x for x in all_ax if x.get_ylabel() == "PFS/OS"][0]
    age_ax = [x for x in all_ax if x.get_ylabel() == "Age"][0]
    time_ax.spines["top"].set_visible(True)
    time_ax.spines["bottom"].set_visible(False)
    time_ax.spines["left"].set_visible(False)
    time_ax.spines["right"].set_visible(True)
    max_os = math.ceil(os.max().values[0] / 5) * 5
    time_ax.set_ylim(0, max_os)
    time_ax.yaxis.tick_right()
    time_ax.invert_yaxis()
    max_age = math.ceil(age.max().values[0] / 5) * 5 + 5
    min_age = math.floor(age.min().values[0] / 5) * 5 - 5
    age_ax.set_ylim(min_age, max_age)
    age_ax.yaxis.tick_right()

    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)

        plt.figure(figsize=(5, 13))
        metadata_annot.plot_legends()
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(add_save_path_suffix(s, "legend", delimiter="_"), dpi=dpi)
        else:
            plt.savefig(
                add_save_path_suffix(save_path, "legend", delimiter="_"), dpi=dpi
            )
    plt.show()


def metadata_boxplot(
    adata,
    sample_col,
    factor1,
    factor2,
    factor1_colors,
    factor2_colors,
    dpi=400,
    save_path=None,
    **kwargs,
):
    adata_lung_patient = get_sample_level_adata(adata, sample_col, feature_columns=None)
    no_val_counts_f1 = adata_lung_patient.obs[factor1].isna().sum()
    no_val_counts_f2 = adata_lung_patient.obs[factor2].isna().sum()

    agg = ObsAggregator(adata_lung_patient, factor1)

    plt.figure(figsize=(3, 4), dpi=dpi)
    agg.get_category_counts(factor2).plot(
        kind="bar", stacked=True, color=factor2_colors.values(), ax=plt.gca(), **kwargs
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()

    plt.figure(figsize=(3, 4))
    agg.get_category_proportions(factor2).plot(
        kind="bar", stacked=True, color=factor2_colors.values(), ax=plt.gca(), **kwargs
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()
    print(f"Filtered out {no_val_counts_f1} samples with no {factor1} value")

    # And inversion
    agg_invert = ObsAggregator(adata_lung_patient, factor2)
    plt.figure(figsize=(3, 4))
    agg_invert.get_category_counts(factor1).plot(
        kind="bar", stacked=True, color=factor1_colors.values(), ax=plt.gca(), **kwargs
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()

    plt.figure(figsize=(3, 4))
    agg_invert.get_category_proportions(factor1).plot(
        kind="bar", stacked=True, color=factor1_colors.values(), ax=plt.gca(), **kwargs
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()

    print(f"Filtered out {no_val_counts_f2} samples with no f{factor2} value")


def metadata_pie(
    adata,
    sample_col,
    factor,
    factor_colors,
    explode_factor=0.05,
    dpi=400,
    save_path=None,
):
    adata_patient = get_sample_level_adata(adata, sample_col, feature_columns=None)
    no_val_counts_bor = adata_patient.obs[factor].isna().sum()
    print(f"Filtered out {no_val_counts_bor} samples with no {factor} value")
    cats = adata_patient.obs[factor].value_counts().index
    cat_counts = adata_patient.obs[factor].value_counts().values
    cat_colors = [factor_colors[cat] for cat in cats]
    explode = [explode_factor for _ in cats]
    wp = {"linewidth": 0.5, "edgecolor": "k"}

    def func(pct, allvalues):
        absolute = int(pct / 100.0 * np.sum(allvalues))
        return f"n = {absolute}"

    fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
    wedges, texts, autotexts = ax.pie(
        cat_counts,
        autopct=lambda pct: func(pct, cat_counts),
        explode=explode,
        labels=cats,
        shadow=True,
        colors=cat_colors,
        startangle=90,
        wedgeprops=wp,
        textprops=dict(color="black"),
    )

    ax.legend(
        wedges, cats, title=factor, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
    )
    plt.setp(autotexts, size=8, weight="bold")
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)
    plt.show()


def filter_out_non_lung_biopsies(adata, original_obs):
    standard_metas = [
        "CPID",
        "cohort",
        "Gender",
        "Age",
        "Smoking_Status",
        "Stage_at_ITx",
        "biopsy_site",
        "histology_type",
    ]
    agg = ObsAggregator(adata, "patient_id")
    print(agg.get_metadata_df("biopsy_site").value_counts())
    adata_lung = adata[adata.obs["biopsy_site"] == "Lung"].copy()
    ls = original_obs.to_list()
    ls.extend(standard_metas)
    ls = list(set(ls))
    adata_lung.obs = adata_lung.obs.loc[:, ls]

    adata_lung = adata_lung[~adata_lung.obs["CB6"].isna()]
    return adata_lung


def filter_out_post_treatment(adata):
    _adata_patient = get_sample_level_adata(adata, "patient_id", feature_columns=None)
    print(_adata_patient.obs["prepost"].value_counts())
    adata_pre = adata[adata.obs["prepost"] == "Pre"].copy()
    return adata_pre


def kaplan_meier_local(
    adata: AnnData,
    event_column: str,
    time_column: str,
    stratifier: str = None,
) -> np.ndarray:
    survival = parse_survival_columns(adata, event_column, time_column)

    results = {}
    if stratifier is not None:
        unique_labels = adata.obs[stratifier].unique()
        # Filter out nas;
        print(
            f"Filtering out NAs for factor {stratifier}: {adata.obs[stratifier].isna().sum()}"
        )
        unique_labels = [x for x in unique_labels if str(x) != "nan"]
        for g in unique_labels:
            g_mask = adata.obs[stratifier] == g
            results[g] = kaplan_meier_estimator(
                survival[g_mask.values]["event"],
                survival[g_mask.values]["time"],
                conf_type="log-log",
            )
    else:
        results["all"] = kaplan_meier_estimator(
            survival["event"], survival["time"], conf_type="log-log"
        )

    stats = None
    if stratifier is not None:
        adata_lung_patient = get_sample_level_adata(
            adata, "patient_id", feature_columns=None
        )
        # Filter out nan stratifiers
        valid_strats = ~adata_lung_patient.obs["CB6"].values.isna()
        # Log-rank test
        stats = compare_survival(
            survival[valid_strats],
            group_indicator=adata.obs[stratifier][valid_strats].values,
            return_stats=True,
        )

    return results, stats


def figure_1c_kaplan_meier(
    adata_lung,
    event_column,
    time_column,
    metadata,
    xlabel,
    ylabel,
    figsize=(10, 7),
    dpi=400,
    save_path=None,
):
    adata_lung_patient = get_sample_level_adata(
        adata_lung, "patient_id", feature_columns=None
    )

    # for k,v in meta_colors.items():
    results = kaplan_meier(
        adata_lung_patient,
        event_column,
        time_column,
        metadata,
    )
    plt.figure(figsize=figsize, dpi=dpi)
    plot_kaplan_meier(
        results,
        with_counts=True,
        stratifier_colors=color_encodings[metadata],
        adata=adata_lung_patient,
        event_column=event_column,
        time_column=time_column,
        stratifier=metadata,
        xlabel=xlabel,
        ylabel=ylabel,
        fill_alpha=0.1,
    )
    plt.title(f"{ylabel} by {metadata}")
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)


def plot_metadata_annotation(
    row_kwargs,
    cell_type_col="cell_types",
    figsize=(5, 15),
    wgap=2,
    plot_legend=False,
    dpi=400,
    save_path=None,
):
    plt.figure(figsize=figsize, dpi=dpi)
    metadata_annot = pch.HeatmapAnnotation(
        axis=0, wgap=wgap, legend=True, orientation="horizontal", **row_kwargs
    )
    metadata_annot.plot_annotations()
    all_ax = plt.gcf().axes
    ct_ax = [x for x in all_ax if x.get_xlabel() == cell_type_col][0]
    ct_ax.set_xlabel("")
    ct_ax.set_xlim([0, 1])
    ct_ax.spines["left"].set_visible(False)
    ct_ax.spines["top"].set_visible(False)
    ct_ax.spines["right"].set_visible(False)

    if len(row_kwargs.keys()) > 1:
        for strat in [x for x in row_kwargs.keys() if x != cell_type_col]:
            or_ax = [x for x in all_ax if x.get_xlabel() == strat][0]
            or_ax.set_xlabel("")

    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)
    plt.show()

    if plot_legend:
        metadata_annot.plot_legends()
        if save_path is not None:
            if isinstance(save_path, list):
                for s in save_path:
                    plt.savefig(
                        add_save_path_suffix(s, "legend", delimiter="_"), dpi=dpi
                    )
            else:
                plt.savefig(
                    add_save_path_suffix(save_path, "legend", delimiter="_"), dpi=dpi
                )


def figure_1d_proportions(
    adata_lung,
    sample_col,
    cell_type_col,
    cell_type_colors,
    cell_type_order,
    sort_by_cell_type,  # Also produces heatmap without this cell type
    exclude_sorted_cell_type=False,
    stratifier_col=None,
    stratifier_colors=None,
    sort_by_stratifier=False,
    figsize=(5, 15),
    hw=1,
    lw=2,
    ec=None,
    dpi=400,
    save_path=None,
):
    aggregator = ObsAggregator(adata_lung, sample_col)
    ct_props = aggregator.get_category_proportions(cell_type_col)
    ct_props.columns = ct_props.columns.droplevel(0)
    ct_props.columns.name = None
    together = ct_props
    if stratifier_col is not None and stratifier_colors is not None:
        ovr = aggregator.get_metadata_df(stratifier_col)
        together = pd.concat([ovr, ct_props], axis=1)

    # Figure 1: Sorted by Cell Type Proportions
    keep_index = together.sort_values(sort_by_cell_type).index
    aggregator_excl = ObsAggregator(
        adata_lung[adata_lung.obs[cell_type_col] != sort_by_cell_type], sample_col
    )
    ct_props_excl = aggregator_excl.get_category_proportions(cell_type_col)
    ct_props_excl.columns = ct_props_excl.columns.droplevel(0)
    ct_props_excl.columns.name = None

    cell_type_colors_excluded = cell_type_colors.copy()
    cell_type_colors_excluded.pop(sort_by_cell_type, None)
    cell_type_order_excluded = cell_type_order.copy()
    cell_type_order_excluded.remove(sort_by_cell_type)

    if stratifier_col is not None and stratifier_colors is not None:
        together_excl = pd.concat([ovr, ct_props_excl], axis=1)

    if stratifier_col is None:
        row_kwargs = {
            cell_type_col: pch.anno_barplot(
                together.sort_values(sort_by_cell_type)[cell_type_order],
                colors=cell_type_colors,
                height=20,
                linewidth=0.5,
                legend=True,
            ),
        }
        plot_metadata_annotation(
            row_kwargs,
            figsize=figsize,
            plot_legend=True,
            dpi=dpi,
            save_path=save_path,
            cell_type_col=cell_type_col,
        )

        if exclude_sorted_cell_type:
            row_kwargs_excl = {
                cell_type_col: pch.anno_barplot(
                    ct_props_excl.loc[keep_index, cell_type_order_excluded],
                    colors=cell_type_colors_excluded,
                    height=20,
                    linewidth=0.5,
                    legend=True,
                ),
            }
            plot_metadata_annotation(
                row_kwargs_excl,
                figsize=figsize,
                plot_legend=True,
                dpi=dpi,
                cell_type_col=cell_type_col,
                save_path=add_save_path_suffix(
                    save_path, f"excluded_{sort_by_cell_type}", delimiter="_"
                ),
            )
    else:
        if sort_by_stratifier:
            together = together.sort_values([stratifier_col, sort_by_cell_type])
            keep_index = together.index
            together_excl = together_excl.loc[keep_index]
        else:
            together = together.loc[keep_index]
            together_excl = together_excl.loc[keep_index]

        row_kwargs = {
            stratifier_col: pch.anno_simple(
                together[stratifier_col],
                colors=stratifier_colors,
                height=hw,
                linewidths=lw,
                edgecolors=ec,
            ),
            cell_type_col: pch.anno_barplot(
                together[cell_type_order],
                colors=cell_type_colors,
                height=20,
                linewidth=0.5,
                legend=True,
            ),
        }
        plot_metadata_annotation(
            row_kwargs,
            figsize=figsize,
            cell_type_col=cell_type_col,
            plot_legend=True,
            dpi=dpi,
            save_path=save_path,
        )

        if exclude_sorted_cell_type:
            row_kwargs_excl = {
                stratifier_col: pch.anno_simple(
                    together_excl[stratifier_col],
                    colors=stratifier_colors,
                    height=hw,
                    linewidths=lw,
                    edgecolors=ec,
                ),
                cell_type_col: pch.anno_barplot(
                    together_excl[cell_type_order_excluded],
                    colors=cell_type_colors_excluded,
                    height=20,
                    linewidth=0.5,
                    legend=True,
                ),
            }
            plot_metadata_annotation(
                row_kwargs_excl,
                figsize=figsize,
                cell_type_col=cell_type_col,
                plot_legend=True,
                dpi=dpi,
                save_path=add_save_path_suffix(
                    save_path, f"excluded_{sort_by_cell_type}", delimiter="_"
                ),
            )


def split_adata_by_tnt(adata):
    adata_nontumor = adata[~adata.obs["cell_types"].isin(["tumorcells"])].copy()
    adata_tumor = adata[adata.obs["cell_types"].isin(["tumorcells"])].copy()
    return adata_nontumor, adata_tumor


def ensure_positive_counts(local_df, marker_level=-1):
    unique_states = set([x[marker_level] for x in local_df.columns])
    if len(unique_states) == 1 and list(unique_states)[0] == 0:
        field = list(local_df.columns[0])
        field[marker_level] = 1
        field = tuple(field)
        local_df[field] = 0
    return local_df


def flatten_mli_columns(df, delim="@", omit_first_field=True):
    new_cols = []
    df = df.copy()
    for c in df.columns:
        if isinstance(c, tuple):
            if omit_first_field:
                new_cols.append(delim.join(c[1:]))
            else:
                new_cols.append(delim.join(c))
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df


def marker_pos_mli_annotate(df, marker_level=-1):
    new_markers = []
    marker_name = df.columns.names[marker_level]
    for c in df.columns:
        c_ls = list(c)
        state = c_ls[-1]
        state_annotated = f"{marker_name}_{state}"
        c_ls[-1] = state_annotated
        new_markers.append(tuple(c_ls))
    df.columns = pd.MultiIndex.from_tuples(new_markers)
    return df


def get_positive_counts_only(
    df,
    marker_level=-1,
):
    marker_name = df.columns.names[marker_level]
    return df.xs(1, level=marker_name, axis=1, drop_level=False)


def cell_type_with_marker_by_region(
    adata,
    cell_type_col="cell_types",
    marker_pos_obsm_key="marker_positivity",
    marker_list=None,
    region_col=None,
    positive_counts_only=True,
):
    master = []
    for c in adata.obs[cell_type_col].unique():
        adata_c = adata[adata.obs[cell_type_col] == c].copy()
        marker_matrix = adata_c.obsm[marker_pos_obsm_key]
        if marker_list is not None:
            marker_matrix = marker_matrix[marker_list]
        adata_c.obs = adata_c.obs.merge(
            adata_c.obsm[marker_pos_obsm_key][marker_list],
            left_index=True,
            right_index=True,
        )
        aggregator = ObsAggregator(adata_c, "patient_id")
        for m in marker_list:
            regions = {}
            if region_col is not None:
                groupings = [region_col, cell_type_col, m]
            else:
                groupings = [cell_type_col, m]
            global local_m
            local_m = aggregator.get_category_counts(groupings)
            local_m = ensure_positive_counts(local_m, marker_level=-1)
            if positive_counts_only:
                local_m = get_positive_counts_only(local_m, marker_level=-1)
            local_m = marker_pos_mli_annotate(local_m, marker_level=-1)
            master.append(local_m)
    master_df_test = pd.concat(master, axis=1)
    master_df_test = master_df_test.fillna(0)
    return master_df_test


def ctx_loop(
    adata,
    cell_type_col,
    region_cols,
    marker_pos_obsm_key,
    marker_list=None,
    positive_counts_only=True,
):
    master_df_by_regions = {}

    # Global;
    counts_by_ct_marker = cell_type_with_marker_by_region(
        adata,  # immune cells
        cell_type_col=cell_type_col,
        marker_pos_obsm_key=marker_pos_obsm_key,
        marker_list=marker_list,
        positive_counts_only=positive_counts_only,
        region_col=None,
    )

    patient_totals = adata.obs.groupby("patient_id").size()
    props_by_ct_marker = counts_by_ct_marker.div(
        patient_totals.loc[counts_by_ct_marker.index], axis=0
    )
    master_df = flatten_mli_columns(props_by_ct_marker)
    master_df_by_regions["global"] = master_df

    # Further stratification
    for r in region_cols:
        counts_by_ct_marker = cell_type_with_marker_by_region(
            adata,  # immune cells
            cell_type_col=cell_type_col,
            marker_pos_obsm_key=marker_pos_obsm_key,
            marker_list=marker_list,
            positive_counts_only=positive_counts_only,
            region_col=r,
        )

        props_by_ct_marker = counts_by_ct_marker.div(
            patient_totals.loc[counts_by_ct_marker.index], axis=0
        )

        props_by_ct_marker = flatten_mli_columns(props_by_ct_marker)
        # annoate with region
        props_by_ct_marker.columns = [f"{r}#{x}" for x in props_by_ct_marker.columns]
        master_df_by_regions[r] = props_by_ct_marker

    return master_df_by_regions


def ctx_loop_normal(
    adata,
    cell_type_col,
    region_cols,
):
    master_df_by_regions = {}
    # Global
    aggregator = ObsAggregator(adata, "patient_id")
    counts_by_ct = aggregator.get_category_proportions(cell_type_col)
    master_df_by_regions["global"] = flatten_mli_columns(counts_by_ct)

    for r in region_cols:
        counts_by_ct_r = aggregator.get_category_proportions(
            [r, cell_type_col], normalisation_column=r
        )
        df = flatten_mli_columns(
            counts_by_ct_r
        )  # different delim for region since @ is used for marker
        df.columns = [f"{r}#{x}" for x in df.columns]
        master_df_by_regions[r] = df
    return master_df_by_regions


def create_within_tnt_proportions(
    adata,
    additional_region_cols,  # tnt_col,
    additional_factor_col,  # strat_col,
    cell_type_col="cell_types",
    func_groups=["immunefunc", "immunemeta", "tumorfunc", "tumormeta"],
    func_markers=None,
):
    adata_nontumor = adata[~adata.obs[cell_type_col].isin(["tumorcells"])].copy()
    adata_tumor = adata[adata.obs[cell_type_col].isin(["tumorcells"])].copy()

    comparators = {}
    for f in func_groups:
        comparators[f] = {}

    subgroups = adata.obs[additional_factor_col].unique()
    subgroups = ["global"] + list(subgroups)

    for funcgroup in func_groups:
        if "immune" in funcgroup:
            adata_subgroup = adata_nontumor
        else:
            adata_subgroup = adata_tumor

        for subgroup in subgroups:
            if subgroup == "global":
                adata_subset = adata_subgroup
            else:
                adata_subset = adata_subgroup[
                    adata_subgroup.obs[additional_factor_col] == subgroup
                ]

            if func_markers is not None:
                comparators[funcgroup][subgroup] = ctx_loop(
                    adata_subset,
                    cell_type_col=cell_type_col,
                    region_cols=additional_region_cols,
                    marker_pos_obsm_key="marker_positivity",
                    marker_list=func_markers[funcgroup],
                    positive_counts_only=True,
                )

            else:
                comparators[funcgroup][subgroup] = ctx_loop_normal(
                    adata_subset,
                    cell_type_col=funcgroup,
                    region_cols=additional_region_cols,
                )

    return comparators


def magnitude_pval_heatmap(
    df,
    p_value_column,
    fc_column,
    feature_field_a_column,
    feature_field_b_column,
    upper_triangle_only=False,
    lower_triangle_only=False,
    triangle_invert_func=(lambda x: -x),
    significance_threshold=0.05,
    fc_abs=1,
    cmap="coolwarm",
    dpi=400,
    figsize=(15, 10),
    save_path=None,
):
    pvals = pd.pivot_table(
        df,
        values=p_value_column,
        index=feature_field_a_column,
        columns=feature_field_b_column,
    )
    pvals = pvals.fillna(1)

    fc = pd.pivot_table(
        df,
        values=fc_column,
        index=feature_field_a_column,
        columns=feature_field_b_column,
    )

    if upper_triangle_only and not lower_triangle_only:
        # Assume square matrix df
        for i in range(fc.shape[0]):
            for j in range(i + 1, fc.shape[1]):
                if pd.isna(fc.iat[i, j]) and not pd.isna(fc.iat[j, i]):
                    fc.iat[i, j] = triangle_invert_func(fc.iat[j, i])
                    fc.iat[j, i] = np.nan
                    pvals.iat[i, j] = pvals.iat[j, i]
                    pvals.iat[j, i] = np.nan

    if lower_triangle_only and not upper_triangle_only:
        # Assume square matrix df
        for i in range(fc.shape[0]):
            for j in range(i + 1, fc.shape[1]):
                if pd.isna(fc.iat[j, i]) and not pd.isna(fc.iat[i, j]):
                    fc.iat[j, i] = triangle_invert_func(fc.iat[i, j])
                    fc.iat[i, j] = np.nan
                    pvals.iat[j, i] = pvals.iat[i, j]
                    pvals.iat[i, j] = np.nan

    pvals_sig = pvals.applymap(lambda x: "*" if x < significance_threshold else "")

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        fc,
        annot=pvals_sig,
        annot_kws={
            "fontsize": 12,
            "color": "k",
            "alpha": 1,
            "va": "center",
            "ha": "center",
        },
        fmt="",
        center=0,
        cmap=cmap,
        vmin=-fc_abs,
        vmax=fc_abs,
    )
    plt.xticks(rotation=90)
    plt.gca().set_aspect("equal")
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)
    plt.show()


def magnitude_pval_volcano(
    df,
    p_value_column,
    fc_column,
    feature_field_a_column,
    feature_field_b_column,
    significance_threshold=0.05,
    log_p_value=False,
    p_max=None,
    fc_abs=1,
    sig_color="red",
    non_sig_color="black",
    title=None,
    figsize=(10, 7),
    feature_sep="\nX\n",
    dpi=400,
    save_path=None,
):
    p_label = f"-log10({p_value_column})" if log_p_value else p_value_column
    p_min = 0
    if p_max is None:
        p_max = df[p_label].max()
        p_max = math.ceil(p_max / 0.5) * 0.5

    absolute_significance_threshold = (
        -np.log10(significance_threshold) if log_p_value else significance_threshold
    )
    if log_p_value:
        df[p_label] = -np.log10(df[p_value_column])

    sig_cols = [
        sig_color if p < significance_threshold else non_sig_color
        for p in df[p_value_column]
    ]

    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x=df[fc_column], y=df[p_label], s=8, c=sig_cols)

    plt.xlabel(fc_column)
    plt.ylabel(p_label)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.axhline(y=absolute_significance_threshold, color="black", linestyle="--")

    plt.xlim([-fc_abs, fc_abs])
    plt.ylim([p_min, p_max])

    # Annotate feature names
    text_x_offset = 0.05
    text_y_offset = 0.01
    for i, row in df.iterrows():
        if row[p_value_column] < significance_threshold:
            plt.annotate(
                row[feature_field_a_column] + feature_sep + row[feature_field_b_column],
                (row[fc_column] + text_x_offset, row[p_label] + text_y_offset),
                fontsize=6,
                color="k",
                alpha=1,
            )
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)
    plt.show()


def mannwhitney_volcano_and_heatmap_by_region(
    adata_lung,
    tnt_dict,
    region_strat,  # must be -> global, nb_tumournontumour_50_2, nb_tumournontumour_20_3, MetaPathNeighbourhood
    save_fig_prefix="figure_1e",
    svg_folder="./",
    png_folder="./",
):
    outer_factor = "global"
    print("=" * 50)
    print(region_strat)
    print("=" * 50)
    comparators = {k: tnt_dict[k][outer_factor][region_strat] for k in tnt_dict.keys()}

    strat = "CB6"
    agg = ObsAggregator(adata_lung, "patient_id")
    strat_df = agg.get_metadata_df(strat)
    strat_cats = strat_df[strat].unique()
    print(strat_df[strat].unique())
    finals = []
    for lab, c in comparators.items():
        patient_adata = ad.AnnData(obs=c)
        patient_adata.obs = patient_adata.obs.fillna(0.0)
        patient_adata.obs = patient_adata.obs.merge(
            strat_df, left_index=True, right_index=True
        )
        # drop cb6 nas
        pseudocount = 1e-5
        patient_adata = patient_adata[~patient_adata.obs[strat].isna()].copy()
        patient_adata.obsm["log"] = patient_adata.obs.iloc[:, :-1].copy()
        patient_adata.obsm["log"] = np.log2(patient_adata.obsm["log"] + pseudocount)
        patient_adata.obsm["log"] = patient_adata.obsm["log"].merge(
            strat_df, left_index=True, right_index=True
        )

        test_results = {}
        dom_results = {}
        mean_results = {}
        log_mean_results = {}
        for feature in c.columns:
            result = univariate_test_feature_by_binary_label(
                patient_adata,
                feature,
                strat,
                parametric=False,  # Mann whitney
            )
            dom = difference_of_means_by_binary_label(
                patient_adata,
                feature,
                strat,
            )  # first - second

            means = patient_adata.obs.groupby(strat)[feature].mean()
            log_means = patient_adata.obsm["log"].groupby(strat)[feature].mean()

            test_results[feature] = result
            dom_results[feature] = dom
            mean_results[feature] = means
            log_mean_results[feature] = log_means

        test_results_df = pd.DataFrame(test_results).T.reset_index()
        test_results_df.columns = ["feature", "statistic", "p_value"]
        dom_results_df = pd.DataFrame(dom_results).T.reset_index(names="feature")
        dom_results_df.columns = ["feature", "diff_means", "diff_ci"]
        means_df = pd.DataFrame(mean_results).T.reset_index(names="feature")
        # means_df.columns = ["feature", "mean_0", "mean_1"]
        log_means_df = pd.DataFrame(log_mean_results).T.reset_index(names="feature")
        # log_means_df.columns = ["feature", "log_mean_0", "log_mean_1"]

        results_df = test_results_df.merge(dom_results_df, on="feature")
        results_df = results_df.merge(means_df, on="feature", suffixes=("", "means"))
        results_df = results_df.merge(
            log_means_df, on="feature", suffixes=("", "_log1p_means")
        )
        results_df["feature_cat"] = lab
        finals.append(results_df)

    finals_df = pd.concat(finals)
    if region_strat == "global":
        finals_df["region"] = "global"
        finals_df["ct"] = finals_df["feature"].apply(lambda x: x.split("@")[0])
        finals_df["marker"] = finals_df["feature"].apply(lambda x: x.split("@")[-1])
    else:
        finals_df["region"] = finals_df["feature"].apply(lambda x: x.split("@")[0])
        finals_df["ct"] = finals_df["feature"].apply(lambda x: x.split("@")[-2])
        finals_df["marker"] = finals_df["feature"].apply(lambda x: x.split("@")[-1])
    finals_df = finals_df.reset_index(drop=True)
    for ix, row in finals_df.iterrows():
        if (row["marker"] == "") or (row["ct"] == row["marker"]):
            finals_df.loc[ix, "marker"] = ""
    finals_df["logFC"] = (
        finals_df[strat_cats[1] + "_log1p_means"]
        - finals_df[strat_cats[0] + "_log1p_means"]
    )  # change order -> so less than 0 indicates upregulate in _0
    finals_df["-log10(p)"] = -np.log10(finals_df["p_value"])
    fc_max = math.ceil(finals_df["logFC"].max())
    fc_min = math.floor(finals_df["logFC"].min())
    fc_abs = max(abs(fc_max), abs(fc_min))
    p_max = finals_df["-log10(p)"].max()
    p_max = math.ceil(p_max / 0.5) * 0.5
    # All features;

    # NOTE: omit any features with 'Other cells'
    finals_df = finals_df[~finals_df["feature"].str.contains("other")]
    # p_max = -np.log10(significance_threshold) if log_p_value else significance_threshold
    # p_max = math.ceil(p_max / 0.5) * 0.5
    for region in finals_df["region"].unique():
        regional_df = finals_df[finals_df["region"] == region]

        magnitude_pval_volcano(
            regional_df,
            p_value_column="p_value",
            fc_column="logFC",
            feature_field_a_column="ct",
            feature_field_b_column="marker",
            significance_threshold=0.05,
            log_p_value=True,
            p_max=p_max,
            fc_abs=fc_abs,
            sig_color="red",
            non_sig_color="black",
            title=f"{region} ({strat}: {strat_cats[0]} vs {strat_cats[1]})",
            save_path=[
                f"{svg_folder}{save_fig_prefix}_volcano_{region}_all_features.svg",
                f"{png_folder}{save_fig_prefix}_volcano_{region}_all_features.png",
            ],
        )

        magnitude_pval_heatmap(
            regional_df,
            p_value_column="p_value",
            fc_column="logFC",
            feature_field_a_column="ct",
            feature_field_b_column="marker",
            significance_threshold=0.05,
            cmap=cmaps.dsmash,
            fc_abs=fc_abs,
            save_path=[
                f"{svg_folder}{save_fig_prefix}_heatmap_{region}_all_features.svg",
                f"{png_folder}{save_fig_prefix}_heatmap_{region}_all_features.png",
            ],
        )

        # By feature subset;
        cats = regional_df["feature_cat"].unique()
        for f in cats:
            sub_df = regional_df[regional_df["feature_cat"] == f]

            magnitude_pval_volcano(
                sub_df,
                p_value_column="p_value",
                fc_column="logFC",
                feature_field_a_column="ct",
                feature_field_b_column="marker",
                significance_threshold=0.05,
                log_p_value=True,
                fc_abs=fc_abs,
                p_max=p_max,
                sig_color="red",
                non_sig_color="black",
                title=f"{region} ({strat}: {strat_cats[0]} vs {strat_cats[1]}), {f}",
                save_path=[
                    f"{svg_folder}{save_fig_prefix}_volcano_{region}_{f}.svg",
                    f"{png_folder}{save_fig_prefix}_volcano_{region}_{f}.png",
                ],
            )

            magnitude_pval_heatmap(
                sub_df,
                p_value_column="p_value",
                fc_column="logFC",
                feature_field_a_column="ct",
                feature_field_b_column="marker",
                significance_threshold=0.05,
                cmap=cmaps.dsmash,
                fc_abs=fc_abs,
                save_path=[
                    f"{svg_folder}{save_fig_prefix}_heatmap_{region}_{f}.svg",
                    f"{png_folder}{save_fig_prefix}_heatmap_{region}_{f}.png",
                ],
            )

        regional_df["p_value_fdr"] = multipletests(
            regional_df["p_value"], method="bonferroni"
        )[1]
        print(regional_df.sort_values("p_value").head(10).to_markdown())
    print("=" * 50)


def plot_continuous_feature_with_median_cutoff(
    adata, event_column, time_column, stratifier, save_path=None, dpi=400, **kwargs
):
    adata = adata.copy()
    adata.obs["temp"] = adata.obs[stratifier] > adata.obs[stratifier].median()
    adata.obs["temp"] = adata.obs["temp"].map(
        {True: f"{stratifier} >= median", False: f"{stratifier} < median"}
    )
    strat_cols = {
        f"{stratifier} >= median": "#7a32da",  # bright purple
        f"{stratifier} < median": "#8f82a1",  # neutral purple
    }
    plot_kaplan_meier(
        kaplan_meier(
            adata,
            event_column,
            time_column,
            "temp",
        ),
        stratifier_colors=strat_cols,
        adata=adata,
        event_column=event_column,
        time_column=time_column,
        stratifier="temp",
        **kwargs,
    )
    plt.title(f"{time_column} by {stratifier}")
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")


def figure_cox_data(
    adata_lung,
    tnt_dict,
    region_strat,
):
    adata_lung_patient = get_sample_level_adata(
        adata_lung, "patient_id", feature_columns=None
    )

    outer_factor = "global"
    print("=" * 50)
    print(region_strat)
    print("=" * 50)
    # comparators = {
    #     "immunefunc": tnt_dict["immunefunc"][outer_factor][region_strat],
    #     "immunemeta": tnt_dict["immunemeta"][outer_factor][region_strat],
    #     "tumorfunc": tnt_dict["tumorfunc"][outer_factor][region_strat],
    #     "tumormeta": tnt_dict["tumormeta"][outer_factor][region_strat]
    # }
    comparators = {k: tnt_dict[k][outer_factor][region_strat] for k in tnt_dict.keys()}

    X = pd.concat(comparators.values(), axis=1).fillna(0.0)

    df = adata_lung_patient.obs.merge(X, left_index=True, right_index=True)
    adata_lung_patient.obs = df

    univariate_results = {}
    for f in X.columns:
        try:
            model = PHReg(
                df["pfs.T"],
                df[f],
                status=df["pfs.E"],
            ).fit()
        except LinAlgError:
            print(f"Failed to fit {f}")
            univariate_results[f] = None
            continue

        summary = model.summary()
        model_meta = summary.tables[0]
        coeff_results = summary.tables[1]
        univariate_results[f] = coeff_results
    results_df = pd.concat(univariate_results.values()).sort_values(by="P>|t|")
    results_df["p_value_fdr"] = multipletests(results_df["P>|t|"], method="bonferroni")[
        1
    ]
    results_df["log_lower"] = np.log(results_df["[0.025"])
    results_df["log_upper"] = np.log(results_df["0.975]"])
    feature_group_map = {}
    for k, v in comparators.items():
        for f in v.columns:
            feature_group_map[f] = k
    results_df["feature_group"] = results_df.index.map(feature_group_map)
    valid = results_df[results_df["log_upper"] != np.inf].reset_index()
    valid = valid[valid["log_lower"] != -np.inf]
    return valid


def figure_forestplot(
    valid,
    figsize_main=(8, 40),
    figsize_fdr=(8, 6),
    save_path=None,
    cutoff=0.05,
    dpi=400,
):
    # First plot; unadjusted p-values
    if cutoff is None:
        data = valid
    else:
        data = valid[valid["P>|t|"] < cutoff]
    # Bounds
    fp.forestplot(
        data,
        varlabel="index",
        estimate="log HR",
        pval="P>|t|",
        ll="log_lower",
        hl="log_upper",
        # model_col="region",
        xlabel="log HR (95% CI)",
        color_alt_rows=True,
        groupvar="feature_group",
        sort=True,
        **{
            "ylabel1_size": 10,
            "xlabel1_size": 100,
            "xtick_size": 14,  # adjust x-ticker fontsize
            "markersize": 50,
        },
        figsize=figsize_main,
        # control size of printed ylabel
    )
    xmin = data["log_lower"].min()
    xmax = data["log_upper"].max()
    xdiff = xmax - xmin
    xscale = xdiff / 8
    plt.xlim([data["log_lower"].min() - xscale, data["log_upper"].max() + xscale])
    if save_path is not None:
        if isinstance(save_path, list):
            for s in save_path:
                plt.savefig(s, dpi=dpi, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Second plot, adjusted;
    data = valid[valid["p_value_fdr"] < 0.05]
    if data.shape[0] != 0:
        fp.forestplot(
            data,
            varlabel="index",
            estimate="log HR",
            pval="p_value_fdr",
            ll="log_lower",
            hl="log_upper",
            # model_col="region",
            color_alt_rows=True,
            groupvar="feature_group",
            xlabel="log HR (95% CI)",
            sort=True,
            **{
                "ylabel1_size": 10,
                "xlabel1_size": 100,
                "xtick_size": 14,  # adjust x-ticker fontsize
                "markersize": 50,
            },
            figsize=figsize_fdr,
            # control size of printed ylabel
        )
        xmin = data["log_lower"].min()
        xmax = data["log_upper"].max()
        xdiff = xmax - xmin
        xscale = xdiff / 8
        plt.xlim([data["log_lower"].min() - xscale, data["log_upper"].max() + xscale])
        if save_path is not None:
            if isinstance(save_path, list):
                for s in save_path:
                    plt.savefig(
                        add_save_path_suffix(s, "fdr_by_feature_group", "_"),
                        dpi=dpi,
                        bbox_inches="tight",
                    )
            else:
                plt.savefig(
                    add_save_path_suffix(save_path, "fdr_by_feature_group", "_"),
                    dpi=dpi,
                    bbox_inches="tight",
                )
    else:
        print("No significant features after FDR correction")


def merge_df_on_patient_adata(patient_level_adata, df):
    df = patient_level_adata.obs.merge(df, left_index=True, right_index=True)
    patient_level_adata.obs = df
    return patient_level_adata


def figure_1f_extended_km(
    adata_lung, all_comparisons, stratifier, dpi=400, save_path=None, **kwargs
):
    # Kaplan meier the signifiant features
    adata_lung_patient = get_sample_level_adata(
        adata_lung, "patient_id", feature_columns=None
    )
    adata_lung_patient = merge_df_on_patient_adata(adata_lung_patient, all_comparisons)

    # adata_lung_patient.obs["cd4tregcells@granzymeb_1"]
    plot_continuous_feature_with_median_cutoff(
        adata_lung_patient,
        "pfs.E",
        "pfs.T",
        stratifier,
        save_path=save_path,
        dpi=dpi,
        **kwargs,
    )


def figure_2_data(
    adata,
):
    colstofill = ["pfs.12months", "pfs.6months", "OS.18months", "OS.24months"]
    for col in colstofill:
        adata.obs[colstofill] = adata.obs[colstofill].fillna(0)

    # Markers
    cols = [
        "asct2",
        "asct2_high",
        "atpa5",
        "atpa5_high",
        "cd11b",
        "cd11c",
        "cd14",
        "cd163",
        "cd19",
        "cd20",
        "cd21",
        "cd31",
        "cd34",
        "cd3e",
        "cd4",
        "cd44",
        "cd44_high",
        "cd45",
        "cd45ro",
        "cd57",
        "cd68",
        "cd8",
        "citratesynthase",
        "citratesynthase_high",
        "col4",
        "cpt1a",
        "cpt1a_high",
        "ecadherin",
        "foxp3",
        "g6pd",
        "g6pd_high",
        "glut1",
        "glut1_high",
        "granzymeb",
        "hexokinase1",
        "hexokinase1_high",
        "hlaa",
        "hlaa_high",
        "icos",
        "idh2",
        "idh2_high",
        "ido1",
        "ido1_high",
        "ki67",
        "lag3",
        "ldha",
        "ldha_high",
        "nakatpase",
        "pancytokeratin",
        "pax5",
        "pd1",
        "pdl1",
        "pdl1_high",
        "pnrf2",
        "pnrf2_high",
        "sdha",
        "sdha_high",
        "sma",
        "vimentin",
    ]
    cols = [c.replace("_", "") for c in cols]

    raw_pos = adata.obsm["marker_positivity"].copy()
    adata.obsm["marker_positivity_nans"] = adata.obsm["marker_positivity"].copy()
    for i, col in enumerate(cols):
        adata.obsm["marker_positivity_nans"][col] = raw_pos[col].replace(
            {"0": np.nan, 0: np.nan, 1: f"{col}"}
        )

    immunefunc = ["pd1", "pdl1", "pdl1_high", "granzymeb", "icos", "ido1", "ido1_high"]
    immunefunc = [c.replace("_", "") for c in immunefunc]
    metafunc = [
        "asct2",
        "asct2_high",
        "atpa5",
        "atpa5_high",
        "citratesynthase",
        "citratesynthase_high",
        "cpt1a",
        "cpt1a_high",
        "g6pd",
        "g6pd_high",
        "glut1",
        "glut1_high",
        "hexokinase1",
        "hexokinase1_high",
        "idh2",
        "idh2_high",
        "nakatpase",
        "pnrf2",
        "pnrf2_high",
        "sdha",
        "sdha_high",
    ]
    metafunc = [c.replace("_", "") for c in metafunc]
    tumourfunc = [
        "pdl1",
        "pdl1_high",
        "vimentin",
        "ki67",
        "ido1",
        "ido1_high",
        "hlaa",
        "hlaa_high",
    ]
    tumourfunc = [c.replace("_", "") for c in tumourfunc]

    adata.obs["cell_types"] = adata.obs["cell_types"].astype("str")

    adata_nontumour = adata[~adata.obs["cell_types"].isin(["tumorcells"])].copy()
    adata_tumour = adata[adata.obs["cell_types"].isin(["tumorcells"])].copy()

    def labelcellpositivity(adata, ctcol, positivitylabels):
        adata = adata.copy()
        adata.obs[ctcol] = (
            adata.obs["cell_types"]
            + "@"
            + adata.obsm["marker_positivity_nans"][positivitylabels]
            .fillna("")
            .agg("".join, axis=1)
        )
        adata.obs[ctcol] = adata.obs[ctcol].astype("category")
        return adata

    adata_nontumour = labelcellpositivity(
        adata=adata_nontumour, ctcol="immune_func_v2", positivitylabels=immunefunc
    )
    adata_nontumour = labelcellpositivity(
        adata=adata_nontumour, ctcol="immune_meta_v2", positivitylabels=metafunc
    )

    adata_tumour = labelcellpositivity(
        adata=adata_tumour, ctcol="tumour_func_v2", positivitylabels=tumourfunc
    )
    adata_tumour = labelcellpositivity(
        adata=adata_tumour, ctcol="tumour_meta_v2", positivitylabels=metafunc
    )

    adata.obs["tumournontumour"] = adata.obs["cell_types"] == "tumorcells"
    adata.obs["tumournontumour"] = adata.obs["tumournontumour"].replace(
        {False: "nontumorcells", True: "tumorcells"}
    )
    adata.obs["tumournontumour"] = adata.obs["tumournontumour"].astype("category")

    # only label cells with functional group if they are more than cut off, else label as base class cell
    def summarisecounts(adata, ctcol, basectcol, cutoff):
        df = pd.DataFrame(adata.obs[ctcol].value_counts())
        df = df[df > cutoff].dropna()
        df.reset_index(inplace=True, names="ct")
        celllist = list(set(df["ct"]))
        adata.obs[ctcol] = adata.obs.apply(
            lambda row: row[ctcol] if row[ctcol] in celllist else row[basectcol], axis=1
        )
        return adata

    ctcols = ["immune_func_v2", "immune_meta_v2"]
    for ctcol in ctcols:
        adata_nontumour = summarisecounts(
            adata=adata_nontumour, ctcol=ctcol, basectcol="cell_types", cutoff=200
        )

    ctcols = ["tumour_func_v2", "tumour_meta_v2"]
    for ctcol in ctcols:
        adata_tumour = summarisecounts(
            adata=adata_tumour, ctcol=ctcol, basectcol="cell_types", cutoff=200
        )

    adata.obs["original_index"] = adata.obs.index
    adata.obs = pd.merge(
        adata.obs,
        adata_nontumour.obs[["uuid", "immune_func_v2"]],
        on="uuid",
        how="left",
    )
    adata.obs = pd.merge(
        adata.obs,
        adata_nontumour.obs[["uuid", "immune_meta_v2"]],
        on="uuid",
        how="left",
    )

    adata.obs = pd.merge(
        adata.obs, adata_tumour.obs[["uuid", "tumour_meta_v2"]], on="uuid", how="left"
    )
    adata.obs = pd.merge(
        adata.obs, adata_tumour.obs[["uuid", "tumour_func_v2"]], on="uuid", how="left"
    )

    adata.obs["immune_func_v2"] = (
        adata.obs[["cell_types", "immune_func_v2"]].ffill(axis=1).iloc[:, -1]
    )
    adata.obs["immune_meta_v2"] = (
        adata.obs[["cell_types", "immune_meta_v2"]].ffill(axis=1).iloc[:, -1]
    )

    adata.obs["tumour_meta_v2"] = (
        adata.obs[["cell_types", "tumour_meta_v2"]].ffill(axis=1).iloc[:, -1]
    )
    adata.obs["tumour_func_v2"] = (
        adata.obs[["cell_types", "tumour_func_v2"]].ffill(axis=1).iloc[:, -1]
    )

    del adata.obsm["marker_positivity_nans"]

    adata.obsm["marker_positivity"].index = adata.obsm[
        "marker_positivity"
    ].index.astype("str")
    # adata.obs.index = adata.obs.index.astype('str')

    # Trim @ if has no marker
    adata.obs["immune_func_v2"] = adata.obs["immune_func_v2"].str.replace(
        r"@$", "", regex=True
    )
    adata.obs["immune_meta_v2"] = adata.obs["immune_meta_v2"].str.replace(
        r"@$", "", regex=True
    )
    adata.obs["tumour_meta_v2"] = adata.obs["tumour_meta_v2"].str.replace(
        r"@$", "", regex=True
    )
    adata.obs["tumour_func_v2"] = adata.obs["tumour_func_v2"].str.replace(
        r"@$", "", regex=True
    )

    adata.obs = adata.obs.set_index("original_index")
    adata.obs = adata.obs.rename_axis(index=None)

    return adata


def cross_proportions(adata, region_col, cell_type_col):
    props = ObsAggregator(adata, ["patient_id", region_col]).get_category_proportions(
        cell_type_col, normalisation_column=region_col
    )

    props = props.fillna(0.0)
    df = props.copy()
    df = df["proportions"]
    patient_ids = df.index.get_level_values("patient_id")
    region_names = df.index.get_level_values(region_col).unique().tolist()
    cell_types = df.columns.tolist()
    region_combinations = list(combinations_with_replacement(region_names, 2))
    cell_type_combinations = list(product(cell_types, repeat=2))
    all_results = {}
    for patient_id in patient_ids:
        df_sub = df.loc[patient_id]
        # region_combinations = list(x for x in combinations_with_replacement(df_sub.index, 2))
        # cell_type_combinations = list(x for x in combinations_with_replacement(df_sub.columns, 2))
        if len(df_sub.index) != len(region_names):
            # find the missing
            missing_regions = set(region_names) - set(df_sub.index)
            for r in missing_regions:
                df_sub.loc[r] = 0.0  # or na

        regions_val = {}
        for r1, r2 in region_combinations:
            if r1 != r2:
                cell_type_combinations = list(product(cell_types, repeat=2))
            else:
                cell_type_combinations = list(
                    combinations_with_replacement(cell_types, 2)
                )
            for c1, c2 in cell_type_combinations:
                r1_c1 = df_sub.loc[r1, c1]
                r2_c2 = df_sub.loc[r2, c2]
                r1_c1_label = f"{r1}@{c1}"
                r2_c2_label = f"{r2}@{c2}"

                if r1_c1 == 0 or r2_c2 == 0:
                    regions_val[(r1_c1_label, r2_c2_label)] = np.nan
                else:
                    regions_val[(r1_c1_label, r2_c2_label)] = r1_c1 / r2_c2

        results = pd.DataFrame(regions_val, index=[patient_id])
        all_results[patient_id] = results

    all_results = pd.concat(all_results, names=["Index"])
    all_results.columns = flatten_mli_cross(all_results.columns, delimiter="%")
    all_results.index = all_results.index.droplevel(1)
    return all_results


def cross_proportion_test(
    adata_lung,
    cross_results,
):
    agg = ObsAggregator(adata_lung, "patient_id")
    strat = "CB6"
    strat_df = agg.get_metadata_df(strat)
    strat_cats = strat_df[strat].unique()
    patient_adata = ad.AnnData(obs=cross_results)
    patient_adata.obs = patient_adata.obs.fillna(0.0)
    patient_adata.obs = patient_adata.obs.merge(
        strat_df, left_index=True, right_index=True
    )
    pseudocount = 1e-5
    patient_adata = patient_adata[~patient_adata.obs[strat].isna()].copy()
    patient_adata.obsm["log"] = patient_adata.obs.iloc[:, :-1].copy()
    patient_adata.obsm["log"] = np.log2(patient_adata.obsm["log"] + pseudocount)
    patient_adata.obsm["log"] = patient_adata.obsm["log"].merge(
        strat_df, left_index=True, right_index=True
    )

    test_results = {}
    dom_results = {}
    mean_results = {}
    log_mean_results = {}
    sample_size = {}
    for feature in cross_results.columns:
        # Custom checks
        # A: If ratio is 0 then ratio doesnt make sense as only measuring one cell type
        # B: if ratio is inf, as above but for denominator
        non_zero = patient_adata.obs[feature] != 0
        non_inf = patient_adata.obs[feature] != np.inf
        valid = non_zero & non_inf
        if (~valid).all():  # If all patients are false, or less than 2 patients total
            print(f"Skipping {feature}")
            continue

        patient_adata_sub = patient_adata[valid].copy()
        if patient_adata_sub.obs[strat].nunique() < 2:
            print(f"Skipping {feature} due to insufficient strata")
            continue

        result = univariate_test_feature_by_binary_label(
            patient_adata_sub,
            feature,
            strat,
            parametric=False,  # Mann whitney
        )
        dom = difference_of_means_by_binary_label(
            patient_adata_sub,
            feature,
            strat,
        )  # first - second

        means = patient_adata_sub.obs.groupby(strat)[feature].mean()
        log_means = patient_adata_sub.obsm["log"].groupby(strat)[feature].mean()

        test_results[feature] = result
        dom_results[feature] = dom
        mean_results[feature] = means
        log_mean_results[feature] = log_means
        sample_size[feature] = dict(patient_adata_sub.obs[strat].value_counts())
    test_results_df = pd.DataFrame(test_results).T.reset_index()
    test_results_df.columns = ["feature", "statistic", "p_value"]
    dom_results_df = pd.DataFrame(dom_results).T.reset_index(names="feature")
    dom_results_df.columns = ["feature", "diff_means", "diff_ci"]
    means_df = pd.DataFrame(mean_results).T.reset_index(names="feature")
    # means_df.columns = ["feature", "mean_0", "mean_1"]
    log_means_df = pd.DataFrame(log_mean_results).T.reset_index(names="feature")
    # log_means_df.columns = ["feature", "log_mean_0", "log_mean_1"]
    sample_size_df = pd.DataFrame(sample_size).T
    sample_size_df.columns = [x + "_counts" for x in sample_size_df.columns]
    sample_size_df = sample_size_df.reset_index(names="feature")

    results_df = test_results_df.merge(dom_results_df, on="feature")
    results_df = results_df.merge(means_df, on="feature", suffixes=("", "means"))
    results_df = results_df.merge(
        log_means_df, on="feature", suffixes=("", "_log1p_means")
    )
    results_df = results_df.merge(sample_size_df, on="feature")
    # finals.append(results_df)
    from statsmodels.stats.multitest import multipletests

    results_df["p_value_fdr"] = multipletests(
        results_df["p_value"], method="bonferroni"
    )[1]
    results_df["logFC"] = (
        results_df[strat_cats[1] + "_log1p_means"]
        - results_df[strat_cats[0] + "_log1p_means"]
    )  # change order -> so less than 0 indicates upregulate in _0
    results_df["-log10(p)"] = -np.log10(results_df["p_value"])

    results_df["ctA"] = results_df["feature"].apply(lambda x: x.split("%")[0])
    results_df["ctB"] = results_df["feature"].apply(lambda x: x.split("%")[-1])
    results_df["regA"] = results_df["ctA"].apply(lambda x: x.split("@")[0])
    results_df["regB"] = results_df["ctB"].apply(lambda x: x.split("@")[0])
    return results_df, strat, strat_cats


def cross_cox(adata, cross_results):
    adata_lung_patient = get_sample_level_adata(
        adata, "patient_id", feature_columns=None
    )
    patient_adata = ad.AnnData(obs=cross_results)
    patient_adata.obs = pd.merge(
        patient_adata.obs,
        adata_lung_patient.obs[["pfs.T", "pfs.E"]],
        left_index=True,
        right_index=True,
    )
    univariate_results = {}
    for feature in cross_results.columns:
        # A: If ratio is 0 then ratio doesnt make sense as only measuring one cell type
        # B: if ratio is inf, as above but for denominator
        non_zero = patient_adata.obs[feature] != 0
        non_inf = patient_adata.obs[feature] != np.inf
        valid = non_zero & non_inf
        if (~valid).all():  # IF all patients are false, skip
            print(f"Skipping {feature}")
            continue

        patient_adata_sub = patient_adata[valid].copy()

        try:
            model = PHReg(
                patient_adata_sub.obs["pfs.T"],
                patient_adata_sub.obs[feature],
                status=patient_adata_sub.obs["pfs.E"],
            ).fit()
        except:  # (LinAlgError, ValueError, IndexError):
            print(f"Failed to fit {feature}")
            univariate_results[feature] = None
            continue

        summary = model.summary()
        model_meta = summary.tables[0]
        coeff_results = summary.tables[1]
        univariate_results[feature] = coeff_results
    results_df = pd.concat(univariate_results.values()).sort_values(by="P>|t|")
    results_df["p_value_fdr"] = multipletests(results_df["P>|t|"], method="bonferroni")[
        1
    ]
    results_df["log_lower"] = np.log(results_df["[0.025"])
    results_df["log_upper"] = np.log(results_df["0.975]"])
    results_df["original_feature_name"] = results_df.index
    # results_df.index = [x[1] for x in results_df.index.str.split("|")]
    results_df = results_df.reset_index()
    valid = results_df[results_df["log_upper"] != np.inf].reset_index()
    return valid


def flatten_mli_cross(mli, delimiter="X"):
    return [f"{k[0]}{delimiter}{k[1]}" for k in mli]


def get_corr(df_long, comp1, comp2, pcb_map):
    r1_c1 = df_long[df_long["concat"] == comp1]
    r2_c2 = df_long[df_long["concat"] == comp2]
    r1_c1_label = comp1
    r2_c2_label = comp2
    r1_c1 = r1_c1[["patient_id", "proportions"]].rename(
        columns={"proportions": r1_c1_label}
    )
    r2_c2 = r2_c2[["patient_id", "proportions"]].rename(
        columns={"proportions": r2_c2_label}
    )
    results = pd.merge(r1_c1, r2_c2, left_on="patient_id", right_on="patient_id")
    results["CB6"] = results["patient_id"].map(pcb_map)

    s1 = results.columns[1]
    s2 = results.columns[2]

    corr_strat = {}
    for r in results["CB6"].unique():
        sub = results[results["CB6"] == r]
        if s1 != s2:
            corr, p = spearmanr(sub[s1], sub[s2])
            corr_strat[(r, "spearman")] = (corr, p)

    return corr_strat


def gcross_univariate(
    gcross_df,
    sample_col,
):
    gcross_df = adata_lung.uns["metabolic_gcross"]["StromaCN"]
    gcross_df.index = gcross_df.index.astype("category")
    gcross_df.index.name = "patient_id"
    # Assuming that gcross_df is a patient X feature dataframe, with index name patient_id

    # Figure 1: Sorted by Cell Type Proportions
    # keep_index = together.sort_values(sort_by_cell_type).index
    # aggregator_excl = ObsAggregator(
    #     adata_lung[adata_lung.obs[cell_type_col] != sort_by_cell_type],
    #     sample_col)
    # ct_props_excl = aggregator_excl.get_category_proportions(cell_type_col)
    # ct_props_excl.columns = ct_props_excl.columns.droplevel(0)
    # ct_props_excl.columns.name = None

    # cell_type_colors_excluded = cell_type_colors.copy()
    # cell_type_colors_excluded.pop(sort_by_cell_type, None)
    # cell_type_order_excluded = cell_type_order.copy()
    # cell_type_order_excluded.remove(sort_by_cell_type)
    adata_lung.uns["metabolic_gcross"]["StromaCN"].index
    ct_props.index


def _build_design_matrix(Y_cond, X_cond, G, neighborhood, ct):
    """Builds a design matrix based on multiple conditional probabilities,
    given a `neighborhood` and `ct`,

    Y_cond -> Y(sample, n, ct | n = neighborhood, ct = ct)
    X_cond -> X(ct | ct = ct)
    G -> G(s | s = s)

    Return dataframe of columns: [Y_cond, X_cond, G]
    with rows as samples with non-na values for all columns (rows or sample
    with missing values are omitted from design matrix).

    """
    G_drop_na = G.dropna()
    Y_cond_cols = Y_cond.columns
    Y_col_sub = Y_cond_cols.get_level_values(level=1).get_loc(ct)
    Y_sub = Y_cond.xs(neighborhood, level=0)[Y_cond_cols[Y_col_sub]]
    # Remove G indices not in Y_sub, if more G indices than Y_sub
    # Take the intersection
    if len(G_drop_na) > len(Y_sub):
        G_drop_na = G_drop_na.loc[Y_sub.index]
    Y_sub = Y_sub.loc[G_drop_na.index]
    Y_sub_remove_nan = Y_sub.dropna()
    # And dont include samples that dont have a value for the response variable
    Y_sub_remove_nan.name = f"Y(sample, ct, n | ct = {ct}, n = {neighborhood})"

    common_index = Y_sub_remove_nan.index
    # Get X(ct = CT), which are the transformed CT frequencies per sample
    X_cond = X_cond.loc[G_drop_na.index]
    X_cond_cols = X_cond.columns
    X_col_sub = X_cond_cols.get_level_values(level=1).get_loc(ct)
    X_sub = X_cond.loc[common_index, X_cond_cols[X_col_sub]]
    X_sub.name = f"X(sample, ct | ct = {ct})"

    # Get G(sample)
    G_sub = G.loc[common_index]
    # Get F(sample) --> all are 1..? NOTE: this maybe redundant then
    F_sub = pd.Series(np.ones(len(G_sub)))
    F_sub.index = common_index
    F_sub.name = "F"
    # contrsuct design matrix for logging
    design_df = pd.concat([Y_sub_remove_nan, X_sub, G_sub, F_sub], axis=1)

    return design_df


def cellular_neighborhood_enrichment(
    adata: AnnData,
    neighborhood: str,
    phenotype: str,
    label: str,
    grouping: str,
    pseudocount: float = 1e-3,
) -> dict:
    """Perform a cellular neighborhood enrichment test using OLS linear models.

    Args:
        adata (AnnData): Annotated data object.
        neighborhood (str): Column in .obs that defines the neighborhood index
            or label that a cell belongs to.
        phenotype (str): Column in .obs that defines the cellular label to take
            into account. Ideally this should be the phenotype that was used to
            compute the given `neighborhood`.
        label (str): Column in .obs that defines the binary label defining
            distinct `grouping` groups.
        grouping (str): Column in .obs that defines distinct samples.
        pseudo_count (float, optional): Pseudocount to add to the log2
            normalised proportions data. Defaults to 1e-3.
    """
    unique_phenotypes = adata.obs[phenotype].unique()
    unique_neighborhoods = adata.obs[neighborhood].unique()
    nhood_agg = ObsAggregator(adata, [neighborhood, grouping])
    sample_agg = ObsAggregator(adata, grouping)
    sample_to_response = sample_agg.get_metadata_df(label)

    # PNC 2D matrix
    ct_props_by_sample = sample_agg.get_category_proportions(phenotype)
    ct_props_by_neighborhood_and_sample = nhood_agg.get_category_proportions(phenotype)

    ct_props_by_sample = ct_props_by_sample.fillna(0.0)
    ct_props_by_neighborhood_and_sample = ct_props_by_neighborhood_and_sample.fillna(
        0.0
    )

    X_sample_log2 = _normalise_log2p(ct_props_by_sample)
    X_sample_neighborhood_log2 = ct_props_by_neighborhood_and_sample.groupby(
        level=0, group_keys=False
    ).apply(_normalise_log2p, pseudocount)

    design_matrices = {}
    p_values = {}
    coefficients = {}
    t_values = {}
    label_encodings = {}

    encoder = LabelEncoder()
    for ct in unique_phenotypes:
        for n in unique_neighborhoods:
            design_df = _build_design_matrix(
                X_sample_neighborhood_log2,
                X_sample_log2,
                sample_to_response,
                n,
                ct,
            )
            Y = design_df.iloc[:, 0]
            X = design_df.iloc[:, 1:]
            X[label] = encoder.fit_transform(X[label].values.reshape(-1, 1))
            regress = sm.OLS(Y, X).fit()
            p_values[(ct, n)] = regress.pvalues[label]
            coefficients[(ct, n)] = regress.params[label]
            t_values[(ct, n)] = regress.tvalues[label]
            design_matrices[(ct, n)] = design_df
            label_encodings[(ct, n)] = dict(enumerate(encoder.classes_))

    p_values_df = _consolidate_statistics(p_values, neighborhood, phenotype)

    coefficients_df = _consolidate_statistics(coefficients, neighborhood, phenotype)

    t_values_df = _consolidate_statistics(t_values, neighborhood, phenotype)

    null_hypothesis, adjusted_pvalues, _, _ = sm.stats.multipletests(
        p_values_df.values.flatten(), method="bonferroni"
    )

    null_hypothesis_df = pd.DataFrame(null_hypothesis.reshape(p_values_df.shape))
    null_hypothesis_df.columns = p_values_df.columns
    null_hypothesis_df.index = p_values_df.index

    adjusted_pvalues_df = pd.DataFrame(adjusted_pvalues.reshape(p_values_df.shape))
    adjusted_pvalues_df.columns = p_values_df.columns
    adjusted_pvalues_df.index = p_values_df.index

    results = {
        "p_values": p_values_df,
        "adjusted_p_values": adjusted_pvalues_df,
        "reject_null_hypothesis": null_hypothesis_df,
        "coefficients": coefficients_df,
        "t_values": t_values_df,
    }

    print(dict(enumerate(encoder.classes_)))
    return results


def cn_enrichment_to_long(enrichment_results_dict):
    dfs = None
    for k, v in enrichment_results_dict.items():
        df = v.copy()
        df_long = df.reset_index().melt(
            id_vars=df.index.name, var_name=df.columns.name, value_name=k
        )
        if dfs is None:
            dfs = df_long
        else:
            dfs = pd.merge(
                dfs, df_long, how="inner", on=[df.index.name, df.columns.name]
            )
    return dfs


def factor_proportions_by_response(
    adata,
    grouping,
    factor,  # Neighborhood key
    response,  # response key
    *,
    inplace=False,
):
    """Computes the proportions (or factors) per sample (grouping) of a given
    factor, between response. Used for neighborhood proportions between
    responses."""
    if pd.api.types.is_numeric_dtype(adata.obs[factor]):
        # create a copy and cast to category
        adata = adata.copy()
        adata.obs[factor] = adata.obs[factor].astype("category")

    sample_helper = ObsAggregator(adata, grouping)
    outer_factor_proportions_by_sample = sample_helper.get_metadata_df(
        factor, skey_handle="category_proportions"
    )

    sample_to_inner_factor = sample_helper.get_metadata_df(
        response
    )  # get the mapping from sample to inner fac

    proportions_master = pd.concat(
        [sample_to_inner_factor, outer_factor_proportions_by_sample], axis=1
    )

    if inplace:
        adata.uns[f"{factor}_proportions_by_{grouping}_between_{response}"]
    else:
        return proportions_master  #


def generate_ytma_figures(
    with_plots=True,
    svg_folder="./",
    png_folder="./",
):
    create_metadata_colors()
    create_cell_type_colors()
    create_cmaps()
    create_meta_path_map()
    adata, original_obs = load_process_input_adata(input_adata_path)

    print("Loaded AnnData input")
    print(f"\t {adata.shape[0]} cells, {adata.obs['patient_id'].nunique()} patients")

    print("Extending with clinical metadata")
    adata = extend_clinical_metadata(
        adata,
        input_clinical_sheet,
        sheet_471=input_clinical_sheet_471,
        sheet_404=input_clinical_sheet_404,
    )

    """ ============================ Figure 1C ============================ """
    if with_plots:
        paths = [
            svg_folder + "figure_1c_clinical_heatmap.svg",
            png_folder + "figure_1c_clinical_heatmap.png",
        ]
        figure_1c_heatmap(adata, save_path=paths)

    print("Filtering out Post-Treatment samples")
    adata = filter_out_post_treatment(adata)
    print(f"\t {adata.shape[0]} cells, {adata.obs['patient_id'].nunique()} patients")
    print("\n")
    print("Filtering out non-lung biopsies")
    adata_lung = filter_out_non_lung_biopsies(adata, original_obs)
    print(
        f"\t {adata_lung.shape[0]} cells, {adata_lung.obs['patient_id'].nunique()} patients"
    )
    # adata_lung.write_h5ad("nsclc_lung_only.h5ad")
    print("\n")
    print("Patients with no CB6:")
    print(adata_lung.obs["CB6"].isna().any())
    _adata_lung_patient = get_sample_level_adata(
        adata_lung, "patient_id", feature_columns=None
    )
    print(_adata_lung_patient.obs["CB6"].value_counts())
    print(_adata_lung_patient.obs["CB6"].value_counts().sum())
    print("\n")
    print(
        f"Filtering out {adata_lung.obs['cell_types'].value_counts()['artifactcells']} artefactual cells"
    )
    adata_lung = adata_lung[adata_lung.obs["cell_types"] != "artifactcells"].copy()
    print(f"\t {adata_lung.shape[0]} cells")

    # _adata_lung_patient = get_sample_level_adata(
    #     adata_lung, "patient_id", feature_columns=None)
    # _adata_lung_patient.obs["cohort"].value_counts()
    if with_plots:
        paths = [
            svg_folder + "figure_1c_cb6_pfs.svg",
            png_folder + "figure_1c_cb6_pfs.png",
        ]
        figure_1c_kaplan_meier(
            adata_lung,
            "pfs.E",
            "pfs.T",
            "CB6",
            "Time (Months)",
            "Overall Survival",
            dpi=400,
            save_path=paths,
        )
    """ =================================================================== """

    """ ============================ Figure 1D ============================ """
    bar_order = [
        #        "artifactcells",
        "othercells",
        "immunenoscells",
        "bcells",
        "plasmacells",
        "cd4tcells",
        "cd4tregcells",
        "cd8tcells",
        "myeloidnoscells",
        # "myeloid_cells",
        "macrophagecells",
        "granulocytecells",
        "endothilialcells",
        "myofibroblastcells",
        "fibroblastcells",
        "tumorcells",
    ]

    # Figure 1D with CB6 stratification
    if with_plots:
        paths = [
            svg_folder + "figure_1d_cb6_stratified.svg",
            png_folder + "figure_1d_cb6_stratified.png",
        ]
        figure_1d_proportions(
            adata_lung,
            "patient_id",
            "cell_types",
            color_encodings["cell_types"],
            bar_order,
            sort_by_cell_type="tumorcells",
            exclude_sorted_cell_type=True,
            sort_by_stratifier=True,
            stratifier_col="CB6",
            stratifier_colors=color_encodings["CB6"],
            figsize=(5, 15),
            hw=1,
            lw=2,
            ec=None,
            dpi=400,
            save_path=paths,
        )
    """ =================================================================== """

    """ ============================ Figure 1xCN ============================ """
    uuid_cn_map = pd.read_csv("./v3/uuid_cn_map.csv", index_col=0)
    # Replace the CNs from this mapping;;
    index = adata_lung.obs.index.copy()
    adata_lung.obs = adata_lung.obs.drop(
        columns=[
            "MetaPathNeighbourhood",
            "nb_tumournontumour_50_2",
            "nb_tumournontumour_20_3",
        ]
    )
    new_obs = adata_lung.obs.merge(uuid_cn_map, on="uuid", how="left")
    new_obs.index = index
    adata_lung.obs = new_obs
    """ =================================================================== """

    """ ============================ GZMB CN ============================ """
    import squidpy as sq
    from napari_prism.models.adata_ops.spatial_analysis._cell_level import (
        cellular_neighborhoods_sq,
    )

    # Run two types of CN:
    adata_lung.obs[["immune_func_v2", "gz"]].groupby("gz").value_counts()["gzmb_pos"]
    adata_lung.obs["gz"] = adata_lung.obsm["marker_positivity"]["granzymeb"]
    adata_lung.obs["gz"] = adata_lung.obs["gz"].map({0: "gzmb_neg", 1: "gzmb_pos"})

    # 1) GZMB+ based CN -> obs.gz
    sq.gr.spatial_neighbors(adata_lung, n_neighs=20)

    cellular_neighborhoods_sq(
        adata_lung,
        phenotype="gz",
        connectivity_key="spatial_connectivities",
        #    library_key: str | None = None,
        k_kmeans=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        mini_batch_kmeans=True,
    )
    inertias = adata_lung.uns["cn_inertias"].reset_index()
    plt.plot(inertias["k_kmeans"], inertias["Inertia"])
    # Go with k = 3,
    adata_lung.obs["gz_cn"] = adata_lung.obsm["cn_labels"]["3"].astype(str)
    sns.heatmap(
        adata_lung.uns["cn_enrichment_matrices"]["3"], cmap="coolwarm", center=0
    )
    adata_lung.obs["gz_cn_merge"] = adata_lung.obs["gz_cn"].map(
        {"0": "GZZ", "1": "non-GZZ", "2": "GZZ"}
    )

    # Viz the CN in suspect regions
    for p in ["31267", "29937"]:
        with plt.rc_context({"figure.figsize": (10, 10)}):
            sc.pl.spatial(
                adata_lung[adata_lung.obs["patient_id"] == p],
                color=["gz", "gz_cn"],
                spot_size=15,
                ncols=2,
                wspace=0.1,
            )
        with plt.rc_context({"figure.figsize": (10, 10)}):
            sc.pl.spatial(
                adata_lung[adata_lung.obs["patient_id"] == p],
                color=["cell_types"],
                palette=color_encodings["cell_types"],
                spot_size=15,
                ncols=2,
                wspace=0.1,
            )

    # 2) CT + GZMB+ based CN ->
    sq.gr.spatial_neighbors(adata_lung, n_neighs=20)
    adata_lung.obs["ct_gz"] = (
        adata_lung.obs["cell_types"].astype(str)
        + "@"
        + adata_lung.obs["gz"].astype(str)
    )
    # 1) GZMB+ based CN -> obs.gz

    cellular_neighborhoods_sq(
        adata_lung,
        phenotype="ct_gz",
        connectivity_key="spatial_connectivities",
        #    library_key: str | None = None,
        k_kmeans=[x for x in range(2, 26)],
        mini_batch_kmeans=True,
    )
    inertias = adata_lung.uns["cn_inertias"].reset_index()
    plt.plot(inertias["k_kmeans"], inertias["Inertia"])
    adata_lung.obs["gz_cn"] = adata_lung.obsm["cn_labels"]["10"].astype(str)
    # adata_lung.obs["gz"] = adata_lung.obs["gz"].astype(int)
    # cell type enrichment
    sns.clustermap(
        adata_lung.uns["cn_enrichment_matrices"]["10"], cmap="coolwarm", center=0
    )
    # Viz the CN in suspect regions
    with plt.rc_context({"figure.figsize": (15, 10)}):
        sc.pl.spatial(
            adata_lung[adata_lung.obs["patient_id"] == "31267"],
            color=["cell_types"],
            spot_size=15,
            ncols=3,
            wspace=0.05,
            palette=color_encodings["cell_types"],
        )
    with plt.rc_context({"figure.figsize": (15, 10)}):
        sc.pl.spatial(
            adata_lung[adata_lung.obs["patient_id"] == "31267"],
            color=["gz_cn", "gz"],
            spot_size=15,
            ncols=3,
            wspace=0.05,
        )

    # sc.pl.spatial?
    # agg.get_category_proportions(["cell_types", "gz"]).sum(axis=0)
    """ ============================ Figure 1X ============================ """
    # Show common signature, lymphocyte/myeloid/structural/tumour in k=2
    immunemyeloid = {
        "bcells": "Lymphocyte",
        "plasmacells": "Lymphocyte",
        "cd4tcells": "Lymphocyte",
        "cd4tregcells": "Lymphocyte",
        "cd8tcells": "Lymphocyte",
        "endothilialcells": "Structural",
        "fibroblastcells": "Structural",
        "granulocytecells": "Myeloid",
        "immunenoscells": "Lymphocyte",
        "macrophagecells": "Myeloid",
        "myeloidnoscells": "Myeloid",
        "myofibroblastcells": "Structural",
        "tumorcells": "Tumour",
        #'artifactcells':'Other',
        "othercells": "Other",
    }
    adata_lung.obs["immunemyeloid"] = (
        adata_lung.obs["cell_types"].map(immunemyeloid).astype("category")
    )

    dd = ctx_loop_normal(adata_lung, "immunemyeloid", ["nb_tumournontumour_50_2"])
    dd = {"immunemyeloid": {"global": dd}}

    dd["immunemyeloid"]["global"]["nb_tumournontumour_50_2"].loc["34697"]
    if with_plots:
        mannwhitney_volcano_and_heatmap_by_region(
            adata_lung,
            dd,
            "nb_tumournontumour_50_2",
            save_fig_prefix="figure_1x_lymphmyeloid",
            svg_folder="svgs/",
            png_folder="pngs/",
        )
    valid = figure_cox_data(adata_lung, dd, "nb_tumournontumour_50_2")
    if with_plots:
        paths = [
            svg_folder + "figure_1x_forestplot_lymphmyeloid.svg",
            png_folder + "figure_1x_forestplot_lymphmyeloid.png",
        ]
        figure_forestplot(valid, cutoff=None, figsize_main=(8, 8), save_path=paths)

    """ =================================================================== """

    """ ============================ Figure 1E ============================ """
    nbs = [
        "nb_tumournontumour_50_2",
        "nb_tumournontumour_20_3",
        "MetaPathNeighbourhood",
    ]

    # TODO: Base cell type comparisons; immune only
    adata_lung_immune = adata_lung[
        ~adata_lung.obs["cell_types"].isin(["tumorcells"])
    ].copy()
    dd = ctx_loop_normal(adata_lung_immune, "cell_types", nbs)
    dd = {"base": {"global": dd}}
    if with_plots:
        mannwhitney_volcano_and_heatmap_by_region(
            adata_lung_immune,
            dd,
            "global",
            save_fig_prefix="figure_1e_base",
            svg_folder=svg_folder,
            png_folder=png_folder,
        )
        for n in nbs:
            mannwhitney_volcano_and_heatmap_by_region(
                adata_lung_immune,
                dd,
                n,
                save_fig_prefix=f"figure_1e_base_{n}",
                svg_folder=svg_folder,
                png_folder=png_folder,
            )

    func_groups = ["immunefunc", "immunemeta", "tumorfunc", "tumormeta"]

    func_markers = {
        "immunefunc": immunefunc_markers,
        "immunemeta": immunemeta_markers,
        "tumorfunc": tumorfunc_markers,
        "tumormeta": tumormeta_markers,
    }
    tnt = create_within_tnt_proportions(
        adata_lung,
        additional_region_cols=nbs,  # tnt_col,
        additional_factor_col="CB6",  # strat_col,
        cell_type_col="cell_types",
        func_groups=func_groups,
        func_markers=func_markers,
    )

    if with_plots:
        mannwhitney_volcano_and_heatmap_by_region(
            adata_lung,
            tnt,
            "global",  # must be -> global, nb_tumournontumour_50_2, nb_tumournontumour_20_3, MetaPathNeighbourhood
            save_fig_prefix="figure_1e_func",
            svg_folder=svg_folder,
            png_folder=png_folder,
        )
        for n in nbs:
            mannwhitney_volcano_and_heatmap_by_region(
                adata_lung,
                tnt,
                n,
                save_fig_prefix="figure_1e_func",
                svg_folder=svg_folder,
                png_folder=png_folder,
            )
    """ =================================================================== """

    """ ============================ Figure 1F ============================ """
    # Then figure 1f
    valid = figure_cox_data(adata_lung, tnt, "global")
    valid["region"] = "global"

    across = []
    for n in nbs:
        valid_n = figure_cox_data(adata_lung, tnt, n)
        valid_n["region"] = [f"{n}#{x[0]}" for x in valid_n["index"].str.split("@")]
        across.append(valid_n)
    valid_rg = pd.concat([valid] + across)
    if with_plots:
        paths = [
            svg_folder + "figure_1f_forestplot_all_regions.svg",
            png_folder + "figure_1f_forestplot_all_regions.png",
        ]
        for p in paths:
            figure_forestplot(valid_rg, figsize_main=(8, 60), save_path=p)

        paths = [
            svg_folder + "figure_1f_forestplot_no_regions.svg",
            png_folder + "figure_1f_forestplot_no_regions.png",
        ]
        for p in paths:
            figure_forestplot(valid, figsize_main=(8, 15), save_path=p)

    # Remerge tnt data flat to patient level
    outer_factor = "global"
    region_comparators = {}
    for r in [
        "global",
        "nb_tumournontumour_50_2",
        "nb_tumournontumour_20_3",
        "MetaPathNeighbourhood",
    ]:
        region_comparators[r] = pd.concat(
            [
                tnt["immunefunc"][outer_factor][r],
                tnt["immunemeta"][outer_factor][r],
                tnt["tumorfunc"][outer_factor][r],
                tnt["tumormeta"][outer_factor][r],
            ],
            axis=1,
        )
    all_comparisons = pd.concat(region_comparators.values(), axis=1)

    # Plot every fdr-passing comparison
    valid_fdr = valid[valid["p_value_fdr"] < 0.05]
    valid_rg_fdr = valid_rg[valid_rg["p_value_fdr"] < 0.05]
    if with_plots:
        for feature in valid_fdr["index"]:
            paths = [
                svg_folder + f"figure_1f_km_{feature}_no_regions.svg",
                png_folder + f"figure_1f_km_{feature}_no_regions.png",
            ]
            figure_1f_extended_km(
                adata_lung,
                all_comparisons,
                feature,
                dpi=400,
                fill_alpha=0.15,  # CI fill alpha for km plot
                with_counts=True,  # Patient counts underneath for each legend cat
                save_path=paths,
            )

        for feature in valid_rg_fdr["index"]:
            paths = [
                svg_folder + f"figure_1f_km_{feature}_all_regions.svg",
                png_folder + f"figure_1f_km_{feature}_all_regions.png",
            ]
            figure_1f_extended_km(
                adata_lung,
                all_comparisons,
                feature,
                dpi=400,
                fill_alpha=0.15,  # CI fill alpha for km plot
                with_counts=True,  # Patient counts underneath for each legend cat
                save_path=paths,
            )
    """ =================================================================== """

    """ ============================ Figure 2B ============================ """
    # Figure 2
    adata_lung = figure_2_data(
        adata_lung
    )  # Multi-marker positivity labels -> immune_func_v2, immune_meta_v2, tumour_func_v2, tumour_meta_v2
    # Repeat the proportion data with new labels; Global Region
    tnt_multipos = create_within_tnt_proportions(
        adata_lung,
        additional_region_cols=nbs,  # tnt_col,
        additional_factor_col="CB6",  # strat_col,
        cell_type_col="cell_types",
        func_groups=[
            "immune_func_v2",
            "immune_meta_v2",
            "tumour_func_v2",
            "tumour_meta_v2",
        ],
        func_markers=None,
    )

    outer_factor = "global"
    region_comparators_multipos = {}
    for r in [
        "global",
        "nb_tumournontumour_50_2",
        "nb_tumournontumour_20_3",
        "MetaPathNeighbourhood",
    ]:
        region_comparators_multipos[r] = pd.concat(
            [
                tnt_multipos["immune_func_v2"][outer_factor][r],
                tnt_multipos["immune_meta_v2"][outer_factor][r],
                tnt_multipos["tumour_func_v2"][outer_factor][r],
                tnt_multipos["tumour_meta_v2"][outer_factor][r],
            ],
            axis=1,
        )

    if with_plots:
        mannwhitney_volcano_and_heatmap_by_region(
            adata_lung,
            tnt_multipos,
            "global",  # must be -> global, nb_tumournontumour_50_2, nb_tumournontumour_20_3, MetaPathNeighbourhood
            save_fig_prefix="figure_2b",
            svg_folder=svg_folder,
            png_folder=png_folder,
        )

        for n in nbs:
            mannwhitney_volcano_and_heatmap_by_region(
                adata_lung,
                tnt,
                n,
                save_fig_prefix="figure_2b",
                svg_folder=svg_folder,
                png_folder=png_folder,
            )

    all_comparisons_multipos = pd.concat(region_comparators_multipos.values(), axis=1)
    """ =================================================================== """

    """ ============================ Figure 2C ============================ """
    valid = figure_cox_data(adata_lung, tnt_multipos, "global")
    valid["region"] = "global"

    across = []
    for n in nbs:
        valid_n = figure_cox_data(adata_lung, tnt_multipos, n)
        valid_n["region"] = [f"{x[0]}" for x in valid_n["index"].str.split("@")]
        across.append(valid_n)
    valid_rg = pd.concat([valid] + across)
    if with_plots:
        paths = [
            svg_folder + "figure_2c_forestplot_all_regions.svg",
            png_folder + "figure_2c_forestplot_all_regions.png",
        ]
        figure_forestplot(valid_rg, figsize_main=(8, 60), save_path=paths)

        paths = [
            svg_folder + "figure_2c_forestplot_no_regions.svg",
            png_folder + "figure_2c_forestplot_no_regions.png",
        ]
        figure_forestplot(valid, figsize_main=(8, 15), save_path=paths)

    # Plot every fdr-passing comparison
    valid_fdr = valid[valid["p_value_fdr"] < 0.05]
    valid_rg_fdr = valid_rg[valid_rg["p_value_fdr"] < 0.05]
    if with_plots:
        for feature in valid_fdr["index"]:
            paths = [
                svg_folder + f"figure_2c_km_{feature}_no_regions.svg",
                png_folder + f"figure_2c_km_{feature}_no_regions.png",
            ]
            figure_1f_extended_km(
                adata_lung,
                all_comparisons_multipos,
                feature,
                dpi=400,
                fill_alpha=0.15,  # CI fill alpha for km plot
                with_counts=True,  # Patient counts underneath for each legend cat
                save_path=paths,
            )

        for feature in valid_rg_fdr["index"]:
            paths = [
                svg_folder + f"figure_2c_km_{feature}_all_regions.svg",
                png_folder + f"figure_2c_km_{feature}_all_regions.png",
            ]
            figure_1f_extended_km(
                adata_lung,
                all_comparisons_multipos,
                feature,
                dpi=400,
                fill_alpha=0.15,  # CI fill alpha for km plot
                with_counts=True,  # Patient counts underneath for each legend cat
                save_path=paths,
            )
    """ =================================================================== """

    """ ============================ Figure 2D/E ============================ """
    # Cross proportions
    for n in nbs:
        cross = cross_proportions(adata_lung, region_col=n, cell_type_col="cell_types")

        cross_results, strat, strat_cats = cross_proportion_test(adata_lung, cross)

        p_max = cross_results["-log10(p)"].max()
        p_max = math.ceil(p_max / 0.5) * 0.5
        fc_max = math.ceil(cross_results["logFC"].max())
        fc_min = math.floor(cross_results["logFC"].min())
        fc_abs = max(abs(fc_max), abs(fc_min))

        inverted_cross_res = cross_results.copy()
        inverted_cross_res["logFC"] = -inverted_cross_res["logFC"]

        if with_plots:
            paths = [
                svg_folder + f"figure_2d_crossprop_volcano_by_{n}.svg",
                png_folder + f"figure_2d_crossprop_volcano_by_{n}.png",
            ]
            magnitude_pval_volcano(
                cross_results,
                p_value_column="p_value",
                fc_column="logFC",
                feature_field_a_column="ctA",
                feature_field_b_column="ctB",
                significance_threshold=0.05,
                log_p_value=True,
                p_max=p_max + 0.5,
                fc_abs=fc_abs,
                sig_color="red",
                non_sig_color="black",
                title=f"{strat}: {strat_cats[1]} vs {strat_cats[0]}",
                feature_sep=" / ",
                save_path=paths,
                dpi=400,
            )

            # Invert CtA and CtB to have values sit on the lower triangle
            # If inverting, hjave to invert the direction
            # Ensure all are on a single diagonal -> added upper_triangle_only param
            paths = [
                svg_folder + f"figure_2d_crossprop_heatmap_by_{n}.svg",
                png_folder + f"figure_2d_crossprop_heatmap_by_{n}.png",
            ]
            magnitude_pval_heatmap(
                inverted_cross_res.sort_values(["ctB", "ctA"]),
                p_value_column="p_value",
                fc_column="logFC",
                feature_field_a_column="ctB",
                feature_field_b_column="ctA",
                significance_threshold=0.05,
                lower_triangle_only=True,
                fc_abs=fc_abs,
                cmap=cmaps.dsmash,
                figsize=(20, 6),
                dpi=400,
                save_path=paths,
            )

        valid = cross_cox(adata_lung, cross)
        valid["feature_group"] = f"cross_proportions_by_{n}"

        if with_plots:
            paths = [
                svg_folder + f"figure_2d_crossprop_forestplot_by_{n}.svg",
                png_folder + f"figure_2d_crossprop_forestplot_by_{n}.png",
            ]
            figure_forestplot(valid, figsize_main=(8, 15), dpi=400, save_path=paths)

        # Correlations matrix;
        n_props = ObsAggregator(adata_lung, ["patient_id", n]).get_category_proportions(
            "cell_types", normalisation_column=n
        )
        n_props = n_props.fillna(0.0)

        df_long = (
            n_props["proportions"]
            .reset_index()
            .melt(
                id_vars=["patient_id", n],
                var_name="cell_types",
                value_name="proportions",
            )
        )
        strat_df = ObsAggregator(adata_lung, "patient_id").get_metadata_df("CB6")
        pcb_map = strat_df.to_dict(orient="index")
        pcb_map = {k: v["CB6"] for k, v in pcb_map.items()}
        df_long["CB6"] = df_long["patient_id"].map(pcb_map)
        df_long["concat"] = df_long[n].astype(str) + "@" + df_long["cell_types"]
        corrs = {}
        for ix, row in cross_results.iterrows():
            corrs[row["feature"]] = get_corr(df_long, row["ctA"], row["ctB"], pcb_map)
        corrs_df = (
            pd.DataFrame(corrs).reset_index().melt(id_vars=["level_0", "level_1"])
        )
        corrs_df["corr"] = corrs_df["value"].apply(lambda x: x[0])
        corrs_df["p"] = corrs_df["value"].apply(lambda x: x[1])
        corrs_df["ctA"] = corrs_df["variable"].apply(lambda x: x.split("%")[0])
        corrs_df["ctB"] = corrs_df["variable"].apply(lambda x: x.split("%")[1])
        yes_corrs_df = corrs_df[corrs_df["level_0"] == "Yes"]
        no_corrs_df = corrs_df[corrs_df["level_0"] == "No"]
        no_corrs_df["variable"] = no_corrs_df["variable"].apply(
            lambda x: x.split("%")[-1] + "%" + x.split("%")[0]
        )
        no_corrs_df["ctA"] = no_corrs_df["variable"].str.split("%").str[0]
        no_corrs_df["ctB"] = no_corrs_df["variable"].str.split("%").str[1]
        corrs_df = pd.concat([yes_corrs_df, no_corrs_df])

        corrcmp = LinearSegmentedColormap.from_list(
            "my_gradient",
            (
                # Edit this gradient at https://eltos.github.io/gradient/#EE9400-EAEAEA-008F78
                (0.000, (0.933, 0.580, 0.000)),
                (0.500, (0.918, 0.918, 0.918)),
                (1.000, (0.000, 0.561, 0.471)),
            ),
        )

        # Omit diagonal
        corrs_df = corrs_df[corrs_df["ctA"] != corrs_df["ctB"]]

        # Lower triangle is the No group
        if with_plots:
            paths = [
                svg_folder + f"figure_2d_crossprop_correlation_heatmap_by_{n}.svg",
                png_folder + f"figure_2d_crossprop_correlation_heatmap_by_{n}.png",
            ]
            magnitude_pval_heatmap(
                corrs_df,
                p_value_column="p",
                fc_column="corr",
                feature_field_a_column="ctA",
                feature_field_b_column="ctB",
                significance_threshold=0.05,
                fc_abs=1,
                cmap=corrcmp,
                dpi=400,
                save_path=paths,
            )
    """ =================================================================== """

    """ ============================ Figure 2+ ============================ """
    # Additional CN-related metrics that could be useful;
    # 1) CN enrichment by cell type, conditional on CB6;
    enr_results = {}
    for c in [
        "cell_types",
        "immune_func_v2",
        "immune_meta_v2",
        "tumour_func_v2",
        "tumour_meta_v2",
    ]:
        for n in nbs:
            enr = cellular_neighborhood_enrichment(
                adata_lung,
                neighborhood=n,
                phenotype=c,
                label="CB6",
                grouping="patient_id",
            )
            enr = cn_enrichment_to_long(enr)
            enr_results[n] = enr

        for n, enr in enr_results.items():
            if (enr["adjusted_p_values"] < 0.05).any():
                print(f"Significant enrichments for {n}")
                print(enr[enr["adjusted_p_values"] < 0.05])
            else:
                print(f"No significant enrichments for {n}")

            if with_plots:
                paths = [
                    svg_folder + f"figure_2p_enrichment_by_{n}.svg",
                    png_folder + f"figure_2p_enrichment_by_{n}.png",
                ]
                magnitude_pval_heatmap(
                    enr,
                    p_value_column="p_values",
                    fc_column="coefficients",
                    feature_field_a_column=c,
                    feature_field_b_column=n,
                    significance_threshold=0.05,
                    fc_abs=abs(enr["coefficients"]).max(),
                    cmap=cmaps.dsmash,
                    dpi=400,
                    figsize=(15, 10),
                    save_path=paths,
                )

    # Compute Jaccard index between parallel CN definitions
    def compute_category_jaccard(df, col1, col2):
        labels1 = df[col1].unique()
        labels2 = df[col2].unique()

        jaccard_matrix = pd.DataFrame(index=labels1, columns=labels2, dtype=float)

        for label1 in labels1:
            for label2 in labels2:
                mask1 = df[col1] == label1
                mask2 = df[col2] == label2
                intersection = (mask1 & mask2).sum()
                union = (mask1 | mask2).sum()
                jaccard_matrix.loc[label1, label2] = (
                    intersection / union if union > 0 else 0
                )

        return jaccard_matrix

    # Compute Jaccard similarity between categories in cn1 vs cn2
    def jaccard_cns(adata, paths):
        metab_order = [
            "Minimal_Metabolic_Activity",
            "Low_Metabolic_Activity",
            "Medium_And_Regulatory_Activity",
            "High_Metabolic_Activity",
        ]
        jaccard_df1 = compute_category_jaccard(
            adata.obs, "nb_tumournontumour_50_2", "nb_tumournontumour_20_3"
        )
        jaccard_df1 = jaccard_df1.sort_index()[
            ["Tumour", "Tumour_Stroma_interface", "Stroma"]
        ]
        sns.heatmap(jaccard_df1, cmap="viridis", vmin=0, vmax=1, annot=True)
        plt.show()

        jaccard_df2 = compute_category_jaccard(
            adata.obs, "nb_tumournontumour_50_2", "MetaPathNeighbourhood"
        )
        jaccard_df2 = jaccard_df2.sort_index()[metab_order]
        sns.heatmap(jaccard_df2, cmap="viridis", vmin=0, vmax=1, annot=True)
        plt.show()

        jaccard_df3 = compute_category_jaccard(
            adata.obs, "nb_tumournontumour_20_3", "MetaPathNeighbourhood"
        )
        jaccard_df3 = jaccard_df3.sort_index()[metab_order]
        sns.heatmap(jaccard_df3, cmap="viridis", vmin=0, vmax=1, annot=True)
        for p in paths:
            plt.savefig(p, dpi=400, bbox_inches="tight")
        plt.show()

    paths = [
        svg_folder + "figure_2p_jaccard_CN.svg",
        png_folder + "figure_2p_jaccard_CN.png",
    ]
    jaccard_cns(adata_lung, paths)
    # Repeat on CB6 subsets;
    paths = [
        svg_folder + "figure_2p_jaccard_CN_CB6Yes.svg",
        png_folder + "figure_2p_jaccard_CN_CB6Yes.png",
    ]
    jaccard_cns(adata_lung[adata_lung.obs["CB6"] == "Yes"], paths)
    paths = [
        svg_folder + "figure_2p_jaccard_CN_CB6No.svg",
        png_folder + "figure_2p_jaccard_CN_CB6No.png",
    ]
    jaccard_cns(adata_lung[adata_lung.obs["CB6"] == "No"], paths)

    # Chi-square test on the CNs; test interdependency of CNs, and between CB6 conditions -> Three-way contigency
    # class_A  = pd.crosstab(
    #     adata_lung[adata_lung.obs["CB6"] == "Yes"].obs["nb_tumournontumour_50_2"],
    #     adata_lung[adata_lung.obs["CB6"] == "Yes"].obs["MetaPathNeighbourhood"]
    # )
    # class_B  = pd.crosstab(
    #     adata_lung[adata_lung.obs["CB6"] == "No"].obs["nb_tumournontumour_50_2"],
    #     adata_lung[adata_lung.obs["CB6"] == "No"].obs["MetaPathNeighbourhood"]
    # )

    # combined = np.array([class_A, class_B])

    # # Sum over classes to get expected row sums
    # expected = combined.sum(axis=0) * (combined.sum(axis=1)[:, None] / combined.sum())
    # from scipy.stats import chi2_contingency

    # # Compute chi-square statistic
    # chi2_stat, p_value, dof, expected_freqs = chi2_contingency(np.vstack([class_A, class_B]))

    # Additional CN stuff;
    def plot_factor_frequencies(
        adata: AnnData,
        grouping: str,  # This represents single points or samples (ideally single Images/ROI)
        factor: str,  # This represents the X-axis; categorical/discrete -> Neighborhoods
        regressor: str,  # This represents the nested categories, usually a response / regressor
        factor_sort: list = None,
        figsize=(15, 8),
        alpha=0.15,
        dodge=0.3,
        dpi=400,
        paths=None,
        **kwargs,
    ):
        """General implementaiton of a strip + point plot for plotting frequencies"""
        # Get the data
        if pd.api.types.is_numeric_dtype(adata.obs[factor]):
            # create a copy and cast to category
            adata = adata.copy()
            adata.obs[factor] = adata.obs[factor].astype("category")

        sample_helper = ObsAggregator(adata, grouping)
        outer_factor_proportions_by_sample = sample_helper.get_metadata_df(
            factor, skey_handle="category_proportions"
        )

        sample_to_inner_factor = sample_helper.get_metadata_df(
            regressor
        )  # get the mapping from sample to inner fac

        proportions_master = pd.concat(
            [sample_to_inner_factor, outer_factor_proportions_by_sample], axis=1
        )

        melted = proportions_master.reset_index().melt(
            id_vars=[grouping, regressor], var_name=f"{factor}", value_name="Proportion"
        )
        melted = melted.sort_values(regressor)
        if factor_sort is not None:
            melted["Category"] = pd.Categorical(
                melted[factor], categories=factor_sort, ordered=True
            )
            melted = melted.sort_values([regressor, "Category"])

        fig, ax = plt.subplots(figsize=figsize)  # TODO: param
        sns.stripplot(
            data=melted,
            x=factor,
            y="Proportion",
            hue=regressor,
            ax=ax,
            alpha=alpha,
            dodge=dodge,
            legend=False,
            **kwargs,
        )
        sns.pointplot(
            data=melted,
            x=factor,
            y="Proportion",
            hue=regressor,
            ax=ax,
            dodge=0.3,
            linestyle="none",
            **kwargs,
        )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if paths is not None:
            for p in paths:
                plt.savefig(p, dpi=dpi, bbox_inches="tight")

        return fig, ax

    nb_order = {
        "nb_tumournontumour_50_2": ["Stroma", "Tumour"],
        "nb_tumournontumour_20_3": ["Stroma", "Tumour_Stroma_interface", "Tumour"],
        "MetaPathNeighbourhood": [
            "Minimal_Metabolic_Activity",
            "Low_Metabolic_Activity",
            "Medium_And_Regulatory_Activity",
            "High_Metabolic_Activity",
        ],
    }
    for n in nbs:
        paths = [
            svg_folder + f"figure_2p_factor_frequencies_by_{n}.svg",
            png_folder + f"figure_2p_factor_frequencies_by_{n}.png",
        ]
        fig, ax = plot_factor_frequencies(
            adata=adata_lung,
            grouping="patient_id",
            factor=n,
            factor_sort=nb_order[n],
            regressor="CB6",
            figsize=(6, 8),
            alpha=0.15,
            dodge=0.3,
            palette=color_encodings["CB6"],
            paths=paths,
        )
        ax.tick_params(axis="x", rotation=90)

        nb_pd = ad.AnnData(
            obs=factor_proportions_by_response(
                adata_lung, "patient_id", n, "CB6"
            ).fillna(0.0)
        )
        for n_s in adata_lung.obs[n].unique():
            result = univariate_test_feature_by_binary_label(
                nb_pd, n_s, "CB6", parametric=False
            )
            print(f"{n}: {n_s}: {result}")
    adata_lung.obs["gz"] = adata_lung.obsm["marker_positivity"]["granzymeb"]
    meta_path_map = label_encodings.meta_path
    return adata_lung


# adata_lung_patient = get_sample_level_adata(adata_lung, "patient_id", feature_columns=None)
adata_lung = generate_ytma_figures(
    with_plots=True, svg_folder="svgs/", png_folder="pngs/"
)

# adata_lung.uns.keys()
# pd.Series(np.array([x for x in adata_lung.uns["metabolicfractions_MetabolicNBs"].columns.str.split("_")])[:, 1]).unique()
# # StromaCN, TumourCN, TumourStromaInterfaceCN
# adata_lung.uns["metabolicfractions_MetabolicNBs"]
# adata_lung
