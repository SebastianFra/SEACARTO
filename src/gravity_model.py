#!/usr/bin/env python
# coding: utf-8
"""Gravity model.
Description
We use the formulation of gravity models to developed trade projections compatible 
with the SSPs. The gravity model coefficients are obtained through a historic regression
based on UNCTAD trade data. Those coefficients are then use to make projections for 
different population and GDP projections aligned with the SSPs.
Projections are made for each ship segment and between any given list of coutries, so
they can used to estimate total transport-work across each pair of countries. For
that, monetary values of trade are translated in to deadweight tonnage and combined
with average distance between countries.
Projections for oil tankers and gas carriers are performed in the next notebook.
Structure
1. Import data and pre-processing
2.  Gravity model estimation
3.  Bilateral trade
"""
# Required packages
from pathlib import Path
import gme as gme
import os
import re
import copy
from sqlite3 import DatabaseError
import sys

sys.path.append(
    Path(__file__).resolve().parents[1]
)  # this line might be necessary for some programming environments

from itertools import product
import pickle
from tkinter import BROWSE

# Scientific packages
import pandas as pd
import numpy as np
import math
from scipy import stats

# Plot and table settings
import matplotlib.pyplot as plt
import seaborn as sns
import csv

plt.style.use("seaborn-notebook")

# Calculations or operations to repeat
redo_calc = {
    "preproc_gm_data": False,
    "proc_all_data": True,
    "gm_estimation": True,
    "bilateral_projection": True,
}
# Base path for file output
input_path = Path(__file__).resolve().parents[1].joinpath("data")

# ## Input parameters
# Years for projection
start_year = 2015
end_year = 2100
step_year = 5
YEARS = np.arange(start_year, end_year + step_year, step_year)


# Scenarios
scenarios = [
    #   "SSP1-19",
    #   "SSP1-26",
    #   "SSP2-45",
    "SSP3-70",
    #   "SSP4-34",
    #   "SSP4-60",
    #   "SSP5-34",
    #   "SSP5-85",
]
SSP = sorted({int(s[3]) for s in scenarios})  # List of SSPs, given in `scenarios`

projection_source = "IIASA GDP"
regressors_gm = [
    "animal_product_demand",
    "gini",
    "governance",
    "fossil_fuel",
    "timber_production",
    "fruit_demand",
    "empty_calories_demand",
    "staples_demand",
    "democracy",
    "rol",
    "fuel_transition",
]
solver = "gm"
cluster = True
iteration = True
if iteration == True:
    with open(
        "/zhome/80/1/163350/maritime_gravity_model/data/others/Scaling_hierarchical.csv"
    ) as EOS_data:
        reader = csv.reader(EOS_data)
        rivalry_list = next(reader)
        rivalry_list.pop(0)
        print(rivalry_list)

else:
    pass
    with open(
        "/zhome/80/1/163350/maritime_gravity_model/data/data/others/EOS_data.csv"
    ) as EOS_data:
        reader = csv.reader(EOS_data)
        rivalry_list = next(reader)
        rivalry_list.pop(0)

# Colors to be use
colors_list = (
    sns.color_palette()
)  # pyam incompatible with python 3.10?? dont accept this change!!
colors_list.remove(
    colors_list[9]
)  # remove scenario not used (SSP index 0-4, SSP-RCP 5-12)
head_num = 5  # number of table lines displayed

# Global variable for data storage/retrieval
COUNTRIES = pd.DataFrame()
DATA = pd.DataFrame()
COEF = dict()
SE = dict()  # Standard errors
p = dict()  # Standard errors
COEF_1 = dict()  # Coefficients
COEF_2 = dict()  # Coefficients
COEF_3 = dict()  # Coefficients
COEF_4 = dict()  # Coefficients
SE_1 = dict()  # Standard errors
SE_2 = dict()  # Standard errors
SE_3 = dict()  # Standard errors
SE_4 = dict()  # Standard errors
p_1 = dict()  # p_values
p_2 = dict()  # p_values
p_3 = dict()  # p_values
p_4 = dict()  # p_values

# Define regressors
INDEP_VARS = {
    "refrigerated cargo_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "liquified gas_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "ro-ro_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "chemical_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "container_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "bulk dry_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
    "oil_value": [
        "distw_log",
        "pop_o_log",
        "pop_d_log",
        "gdp_d_log",
        "gdp_o_log",
        "democracy_o",
        "democracy_d",
        "intercept",
    ],
}
INDEP_VARS_1 = INDEP_VARS
INDEP_VARS_2 = INDEP_VARS
INDEP_VARS_3 = INDEP_VARS
INDEP_VARS_4 = INDEP_VARS


def read_data(YEARS_HIST):
    """Import data and pre-processing. (1)
    Evaluate correlation between parameters and confirms the relation between trade and
    GDP, hence GDP growth in the next steps can be used as a proxy for growth in
    bilateral trade.
    Data is downloaded from http://www.cepii.fr/CEPII (HS and gravity datasets), among
    them:
    - Population-weighted distance between most populated cities (km)
    - Trade flow (in thousands current USD) (source: BACI)
    - GDP (current thousands USD)
    - Population (in millions)
    This data set is then merged with additional datasets that are now used as part of
    the gravity model. Additional datasets, include at the moment, oil price in dollars,
    trade costs provided by ESCAP and trade by ship segments in velue and weight
    provided by BACI.
    """
    global COUNTRIES, DATA, COUNTRIES_num

    # ### 1.1 Filter rows and columns
    # Import gravity model data set from CEPII website (
    # http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele.asp). Rows with null values are
    # removed and log columns are created for parameters used in the gravity model.

    # import CEPII data
    data_CEPII = pd.read_csv(
        input_path / "gravity_model" / "CEPII_v202102.csv", low_memory=False
    )
    COUNTRIES = pd.read_csv(
        input_path / "others" / "countries_group_code_medium.csv",
        dtype={"iso_code": int, "iso_code_uncom_o": int, "iso_code_uncom_d": int},
        encoding="utf-8-sig",
    )
    COUNTRIES_num = len(COUNTRIES)
    # filter negative and null vales and create log
    for name in [
        "gdp_ppp_o",
        "gdp_ppp_d",
        "pop_o",
        "pop_d",
        "distw",
    ]:
        data_CEPII = data_CEPII[data_CEPII[name] > 0]
        data_CEPII[f"{name}_log"] = np.log(data_CEPII[name])
        data_CEPII["gdp_ppp_log"] = np.log(
            data_CEPII["gdp_ppp_o"] * data_CEPII["gdp_ppp_d"]
        )

    # filter countries selected in the analysis
    for filter in ["iso3num_o", "iso3num_d"]:
        data_CEPII = data_CEPII[data_CEPII[filter].isin(list(COUNTRIES["iso_code"]))]

    # Filter years to be used. Years previous to 1995 are removed, they may not be
    # representative of currents trade patterns
    DATA = data_CEPII[data_CEPII["year"].isin(YEARS_HIST)]
    # merge the BACI iso codes to the DATA file
    DATA = DATA.merge(
        COUNTRIES.loc[
            :,
            [
                "iso_code_alpha_o",
                "iso_code_uncom_o",
            ],
        ],
        left_on=["iso3_o"],
        right_on=["iso_code_alpha_o"],
        how="left",
    )
    DATA = DATA.merge(
        COUNTRIES.loc[
            :,
            [
                "iso_code_alpha_d",
                "iso_code_uncom_d",
            ],
        ],
        left_on=["iso3_d"],
        right_on=["iso_code_alpha_d"],
        how="left",
    )


def preproc_gm_data(ship_types: list, YEARS_HIST: np.array, output_path: str):
    """Merge all data sets in one. (1.2)
    Aggregate all data in one data frame. As an alterantive for standard CEPII trade
    indicators, we use BACI data to parse commodities in specific ship segments by value
    and weight. This follows the methodology and segment division established by "Wang,
    Xiao-Tong, et al. "Trade-linked shipping CO2 emissions." Nature Climate Change 11.11
    (2021): 945-951."
    """
    # get baci trade in monetary and product codes HS are obtained from Wang 2021
    # http://www.cepii.fr/DATA_DOWNLOAD/baci/doc/DescriptionBACI.html
    products_data = pd.read_csv(
        input_path / "baci" / "product_codes_HS17_HS92_shiptype_short_adj.csv"
    )

    # function to sum all commodities trade between a pair of countries
    def sum_exports(x, baci_year, value_or_weight):
        temp = baci_year.loc[
            (baci_year["i"] == int(x["iso_code_uncom_o"]))
            & (baci_year["j"] == int(x["iso_code_uncom_d"]))
        ]
        temp = temp.apply(pd.to_numeric, errors="coerce")
        temp = temp.dropna()
        temp.to_csv(output_path / "dataframe" / "temp.csv")
        temp["q"] = temp.q.astype(float)
        try:
            val = float(np.array(temp[value_or_weight]).sum())
            return val
        except ValueError:
            print("temp has NA")

    # return BACI across countries, years, and ship types
    ship_types_short = [
        "container_value",
        "chemical_value",
        "refrigerated cargo_value",
        "ro-ro_value",
        "oil_value",
        "bulk dry_value",
    ]
    for ship in ship_types_short:
        DATA[f"{ship}_value"], DATA[f"{ship}_weight"] = 0, 0
        list_prod = list(
            products_data.loc[(products_data["ship type"] == f"{ship}")]["hs92"]
        )
        for year in YEARS_HIST:
            print(year)
            baci_year = pd.read_csv(
                "/zhome/80/1/163350/maritime_gravity_model/data/baci/BACI_HS92_Y"
                + str(year)
                + "_V202201.csv"
            )
            baci_year = baci_year.loc[baci_year["k"].isin(list_prod)]
            for i, j in zip(["_value", "_weight"], ["v", "q"]):
                DATA[ship + i] = DATA.apply(
                    lambda x: x[ship + i]
                    if x["year"] != year
                    else sum_exports(x, baci_year, j),
                    axis=1,
                )

    DATA.to_csv(os.path.join(output_path, "dataframe", "CEPII_preproc.csv"))


def ssp_data(regressors_gm: str, output_path: str) -> None:
    global DATA
    # load the SSP data for future projections
    for reg in regressors_gm:
        # special treatment for this domestic only variable
        if reg == "domestic_area":
            x = "o"
            temp = pd.read_csv(
                input_path / "SSP" / f"{reg}_historic.csv"
            )  # , on_bad_lines="skip")
            DATA = pd.merge(DATA, temp, how="left", on=["year", f"iso3_{x}"])
            DATA.loc[
                DATA.iso3_o == DATA.iso3_d, f"domestic_area_{x}"
            ] = 0.0  # set value for all non domestic routes to 0
        else:
            try:
                for x in ["o", "d"]:
                    temp = pd.read_csv(
                        input_path / "SSP" / f"{reg}_historic_{x}.csv"
                    )  # , on_bad_lines="skip")
                    temp[f"{reg}_{x}_log"] = np.log(temp[f"{reg}_{x}"])
                    DATA = pd.merge(DATA, temp, how="left", on=["year", f"iso3_{x}"])
            except Exception as e:
                raise ValueError(
                    "Are you sure you added the fight components to the regressors_gm list???"
                )
    DATA["dist**2"] = np.power(DATA["dist"], 2)
    DATA["dist**-2"] = 1 / np.power(DATA["dist"], 2)
    DATA.to_csv(os.path.join(output_path, "dataframe", "intermediate_data.csv"))


def build_cluster():
    # split data frame according to the clustering
    clustering = pd.read_csv(input_path / "others" / "cluster_regimes_4.csv")

    DATA["o_d"] = DATA["iso3_d"] + DATA["iso3_o"]
    cluster_1 = clustering["cluster"] == 1
    cluster_2 = clustering["cluster"] == 2
    cluster_3 = clustering["cluster"] == 3
    cluster_4 = clustering["cluster"] == 4
    # filter
    clustering_1 = clustering[cluster_1]
    clustering_2 = clustering[cluster_2]
    clustering_3 = clustering[cluster_3]
    clustering_4 = clustering[cluster_4]
    # safe globally
    global DATA_1
    global DATA_2
    global DATA_3
    global DATA_4
    global DATA_5
    # create cluster specific DFs
    DATA_1 = pd.merge(DATA, clustering_1, how="right", on=["o_d"])
    DATA_2 = pd.merge(DATA, clustering_2, how="right", on=["o_d"])
    DATA_3 = pd.merge(DATA, clustering_3, how="right", on=["o_d"])
    DATA_4 = pd.merge(DATA, clustering_4, how="right", on=["o_d"])
    DATA_5 = DATA

    DATA_1 = pd.DataFrame(DATA_1)
    DATA_2 = pd.DataFrame(DATA_2)
    DATA_3 = pd.DataFrame(DATA_3)
    DATA_4 = pd.DataFrame(DATA_4)
    DATA_5 = pd.DataFrame(DATA_5)

    return (DATA_1, DATA_2, DATA_3, DATA_4, DATA_5)


def estimate_gm(ship_type, c):
    """Gravity Model. (2)
    This function implements the structural GE gravity model. The model applies GDP
    projection of individual countries for all five SSP scenarios. Data is based on
    previously filtered CEPII data and IIASA-OECD projections.
    Other variables could be included in the gravity model, i.e., "contig",
    "comlang_off", "comcol", "rta", "comrelig", "pop_o", "pop_d", "gdpcap_o", "gdpcap_d"
    providing small gains in R2 contig: 0.001, comlang_off: 0.004, comcol: 0.003,
    rta: 0.000, comrelig: 0.002, pop: 0.000, gdpcap: 0.000. The main issue is that most
    of these variables are binary (0 or 1) and do not fit really well into the
    regression model. Shepherd et al. (2019) succeed somehow in implementing some of
    these variables but with a significantly smaller R2 of 0.54.
    $$ T_{ij} = A \frac{GDP_i \times GDP_j }{D_{ij}}$$
    Where $T$ represents bilateral trade and $D$ is ditance, both are given between
    countries $i$ and $j$.
    Sources:
    - Yotov, Piermartini, Monteiro, and Larch (2016) - An Advanced Guide to Trade Policy
      Analysis: The Structural Gravity Model.
    - Shepherd, Doytchinova, and Kravchenko (2019) - The gravity model of international
      trade: a user guide [R version].
    - Package gegravity - https://peter-herman.github.io/gegravity/#example
    """
    # ### 2.1 Setting-up and running the model the model
    # Prepare data and econometric inputs for the GE Model using the tools in the gme
    # package. First, Define a gme EstimationData object.\
    # Apart from trade flow BACI, other options available include trade flow of
    # manufactured goods.
    # Trade is given in thousand dollars.
    global SE
    global p
    global COEF
    DATA = globals()[f"DATA_{c}"]
    DATA["intercept"] = 1
    INDEP_VARS = globals()[f"INDEP_VARS_{c}"]
    # filter for beta intervals
    gm_years = np.arange(2003, 2018)
    DATA = DATA.loc[DATA["year"].isin(gm_years)]
    DATA.to_csv(output_path / "dataframe" / "DATA_before_esstimation.csv")
    # Add a constant column
    # TODO consolidate this with the initial creation of `DATA`

    gme_data = gme.EstimationData(
        data_frame=DATA,
        imp_var_name="iso3_d",
        exp_var_name="iso3_o",
        trade_var_name=ship_type,
        year_var_name="year",
    )

    # statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF = False
    augmented_model = gme.EstimationModel(
        estimation_data=gme_data,
        # left-hand side are not log, the package converts it automatically
        lhs_var=ship_type,
        rhs_var=INDEP_VARS[ship_type],
        # fixed_effects=[["iso3_d"], ["iso3_o"]],
        # std_errors="HC1",  # i.e. heteroskedasticity-consistent
    )

    with HiddenPrints():
        estimates = augmented_model.estimate()

    # Store results for write tex files with the regression tables
    results = estimates["all"]
    print(ship_type, results.summary())
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}"""
    endtex = "\end{document}"
    # f = open(ship_type + '_results.txt')
    with open(
        os.path.join(
            f"/zhome/80/1/163350/maritime_gravity_model/outputs/{ship_type}_{c}_results.tex"
        ),
        "w",
    ) as f:
        f.write(beginningtex)
        f.write(results.summary2().as_latex())
        f.write(endtex)
        f.close()
    p[ship_type] = results.pvalues
    COEF[ship_type] = results.params
    SE[ship_type] = results.bse
    ret = [COEF, SE, p, results]
    ret.append(augmented_model)
    return ret


def exo_data_for_projection(YEARS, ship_types) -> pd.DataFrame:
    """Add variables bilateral data for gravity model. (3.2)
    Create a list with columns of gdp and population for every year that will be addeed
    to the main dataset. For every pair of the country, we interpolate result from IAMC
    database to match the desired values.
    This part of the code is not fully parametrized yet, so it has to be updated every
    variable in the gravity model are updated.
    """

    data, params, columns_names, ind_params_adj, dep_params_adj = [], [], [], [], []
    # get original DATA
    # get a list of unique parameters used for all ship types
    for type in ship_types:
        if cluster == True:
            params.extend((list(COEF_1[type].keys())))
        else:
            params.extend((list(COEF[type].keys())))
    for param in params:
        x = param.split("_log")[0]
        if len(x.split("_")) > 1 and x.split("_")[-1] in ["o", "d"]:
            dep_params_adj.append(x)
        else:
            ind_params_adj.append(x)
    # get a multi-index pandas with parameters
    # get ride of duplicates
    ind_params = []
    for i in ind_params_adj:
        if i not in ind_params:
            ind_params.append(i)
    dep_params = []
    for i in dep_params_adj:
        if i not in dep_params:
            dep_params.append(i)
    ssp_data_csv = {}
    for param in dep_params:
        short = param.split("_")[0]
        if os.path.isfile(input_path / "SSP" / f"{short}_future.csv"):
            ssp_data_csv[short] = pd.read_csv(
                input_path / "SSP" / f"{short}_future.csv"
            )
        else:
            ssp_data_csv[short] = pd.read_csv(
                input_path / "SSP" / f"gdp_future.csv",
                usecols=["Model", "Scenario", "Region", "Variable"],
            )  # copy layout
            r = (
                DATA.loc[:, ["iso3_o", param]]
                .drop_duplicates("iso3_o")
                .set_index("iso3_o")
            )
            ssp_data_csv[short] = ssp_data_csv[short].merge(
                r, left_on="Region", right_index=True
            )
            ssp_data_csv[short]["Variable"] = short
            for y in YEARS:
                ssp_data_csv[short].loc[:, str(y)] = ssp_data_csv[short][param]

    for ssp, i, j in product(SSP, COUNTRIES.index, COUNTRIES.index):
        if i == j:
            continue  # no default skipping of domestic transportcontinue

        # one row for each combination
        data_row = []
        # time independent parameters
        values = []
        DATA["iso3num_o"] = DATA["iso3num_o"].astype("int")
        DATA["iso3num_d"] = DATA["iso3num_d"].astype("int")
        for param in ind_params:
            distance = DATA.loc[
                (DATA["iso3num_o"] == int(COUNTRIES.loc[j, "iso_code"]))
                & (DATA["iso3num_d"] == int(COUNTRIES.loc[i, "iso_code"]))
            ]
            try:
                values.append(float(distance[param].values[-1]))
            except:
                pass

        data_row.extend(
            [
                COUNTRIES.loc[i, "iso_code"],
                COUNTRIES.loc[j, "iso_code"],
                COUNTRIES.loc[i, "iso_code_alpha"],
                COUNTRIES.loc[j, "iso_code_alpha"],
                ssp,
                *values,
            ]
        )

        variable_mapper = {
            "gdp": "GDP|PPP",
            "pop": "Population",
            "democracy": "Democracy",
            "rol": "Democracy",
        }
        for param in dep_params:
            short = param.split("_")[0]
            param_pandas = ssp_data_csv[short]
            var = variable_mapper[short] if short in variable_mapper.keys() else short
            param_pandas = param_pandas.loc[
                (param_pandas["Scenario"] == "SSP" + str(ssp))
                & (param_pandas["Model"] == projection_source)
                & (param_pandas["Variable"] == var)
            ]
            param_pandas.to_csv(output_path / "dataframe" / "param_pandas_test.csv")
            param_pandas["Region"] = param_pandas["Region"].astype("str")
            COUNTRIES["iso_code_alpha"] = COUNTRIES["iso_code_alpha"].astype("str")
            temp = param_pandas.loc[
                param_pandas["Region"]
                == COUNTRIES.loc[
                    np.where(param.split("_")[1] == "o", i, j), "iso_code_alpha"
                ]
            ]
            temp = temp.drop_duplicates(subset="Model", keep="first")
            # assert len(temp)>0, f'Check if variable {var} in {variable_mapper} because not in {param_pandas.columns}'
            for year in YEARS:
                try:
                    data_row.append(
                        float(
                            temp[str(year)].values
                            * np.where(short == "gdp", 1e6, 1)
                            * np.where(short == "pop", 1000, 1)
                        )
                    )
                except:
                    pass

        data.append(data_row)

    data_test = pd.DataFrame(data)
    data_test.to_csv(output_path / "dataframe" / "data_df.csv")
    # format in a pandas dataframe
    for param in ind_params:
        columns_names.append(param)
    for param in dep_params:
        for year in YEARS:
            columns_names.append(f"{year}_{param}")
    global bilateral_data
    bilateral_data = pd.DataFrame(
        data, columns=["export", "import", "iso3_o", "iso3_d", "ssp"] + columns_names
    )
    bilateral_data.to_csv(output_path / "dataframe" / "bilateral_data_b4drop.csv")
    bilateral_data.replace("", np.nan, inplace=True)
    bilateral_data.dropna(inplace=True)
    bilateral_data.to_csv(output_path / "dataframe" / "bilateral_data_afterdrop.csv")
    return bilateral_data


def bilateral_data_cluster():
    # split data frame according to the clustering
    clustering = pd.read_csv(input_path / "others" / "cluster_regimes_4.csv")
    bilateral_data["o_d"] = bilateral_data["iso3_d"] + bilateral_data["iso3_o"]
    bilateral_data
    bilateral_data.to_csv(
        output_path / "dataframe" / "bilateral_data_test_b4clustering.csv"
    )
    cluster_1 = clustering["cluster"] == 1
    cluster_2 = clustering["cluster"] == 2
    cluster_3 = clustering["cluster"] == 3
    cluster_4 = clustering["cluster"] == 4
    # filter
    clustering_1 = clustering[cluster_1]
    clustering_2 = clustering[cluster_2]
    clustering_3 = clustering[cluster_3]
    clustering_4 = clustering[cluster_4]
    # safe globally
    global bilateral_data_1
    global bilateral_data_2
    global bilateral_data_3
    global bilateral_data_4
    global bilateral_data_5

    # create cluster specific DFs
    bilateral_data_1 = pd.merge(bilateral_data, clustering_1, how="right", on=["o_d"])
    bilateral_data_2 = pd.merge(bilateral_data, clustering_2, how="right", on=["o_d"])
    bilateral_data_3 = pd.merge(bilateral_data, clustering_3, how="right", on=["o_d"])
    bilateral_data_4 = pd.merge(bilateral_data, clustering_4, how="right", on=["o_d"])
    bilateral_data_5 = bilateral_data

    bilateral_data_1 = pd.DataFrame(bilateral_data_1)
    bilateral_data_2 = pd.DataFrame(bilateral_data_2)
    bilateral_data_3 = pd.DataFrame(bilateral_data_3)
    bilateral_data_4 = pd.DataFrame(bilateral_data_4)
    bilateral_data_5 = bilateral_data_5
    bilateral_data_1.to_csv(output_path / "dataframe" / "bilateral_data_test_1.csv")
    bilateral_data_2.to_csv(output_path / "dataframe" / "bilateral_data_test_2.csv")
    bilateral_data_3.to_csv(output_path / "dataframe" / "bilateral_data_test_3.csv")

    return (bilateral_data_1, bilateral_data_2, bilateral_data_3, bilateral_data_4)


def get_var_names() -> None:
    Segment_parameters = pd.read_csv(
        input_path / "SSP" / "segment_parameters.csv", low_memory=False, index_col=[0]
    )


def run_gm_cluster(r, ship_type, c, bilateral_data, YEARS_HIST, output_path):
    """Run gravity model. (3.4)
    Use gravity model coefficients to estimate trade between countries. The gravity
    model is run using interpolated data for historic data (hindcast analysis) and
    projections (SSP projections), depending on the coefficients chosen. It can be run
    for trade (all shipping sectors) or/and shipping sectors individually.
    """
    Scaling_data = pd.read_csv(
        os.path.join(input_path, "others", "Scaling_hierarchical.csv")
    )
    Scaling_data = Scaling_data.set_index("Scaling", drop=False)
    Scaling = Scaling_data.loc[f"Scaling_{ship_type}_{c}", f"{r}"]
    grav_model_coeff = {}
    grav_model_coeff = copy.deepcopy(globals()[f"COEF_{c}"][ship_type])
    param_nolog, param_log, param_gm_log, param_gm_nolog = [], [], [], []
    for param in grav_model_coeff.index:
        if len(param.split("_log")) == 2:
            inter = param.split("_log")[0]
            if inter.split("_")[-1] in ["o", "d"]:
                param_log.append(param)
            else:
                param_gm_log.append(param)
        elif param.split("_")[-1] in ["o", "d"]:
            param_nolog.append(param)
        else:
            param_gm_nolog.append(param)
    if c == 1:
        for year in YEARS:
            bilateral_data["trade_" + ship_type + str(year)] = bilateral_data.apply(
                lambda x: np.exp(
                    (grav_model_coeff["intercept"])
                    + (grav_model_coeff["pop_o_log"] * np.log(x[f"{year}_pop_o"]))
                    + (grav_model_coeff["pop_d_log"] * np.log(x[f"{year}_pop_d"]))
                    + (grav_model_coeff["gdp_o_log"] * np.log(x[f"{year}_gdp_o"]))
                    + (grav_model_coeff["gdp_d_log"] * np.log(x[f"{year}_gdp_d"]))
                    + (grav_model_coeff["distw_log"] * np.log(x["distw"]))
                    + (
                        grav_model_coeff["democracy_o"]
                        + (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_o"]
                    + (
                        grav_model_coeff["democracy_d"]
                        + (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_d"]
                ),
                axis=1,
            )
        bilateral_data.fillna(0, inplace=True)
        bilateral_data.to_csv(output_path / "dataframe" / f"bilateral_data_{c}_{r}.csv")
        return bilateral_data
    if c == 2:
        for year in YEARS:
            bilateral_data["trade_" + ship_type + str(year)] = bilateral_data.apply(
                lambda x: np.exp(
                    (grav_model_coeff["intercept"])
                    + (grav_model_coeff["pop_o_log"] * np.log(x[f"{year}_pop_o"]))
                    + (grav_model_coeff["pop_d_log"] * np.log(x[f"{year}_pop_d"]))
                    + (grav_model_coeff["gdp_o_log"] * np.log(x[f"{year}_gdp_o"]))
                    + (grav_model_coeff["gdp_d_log"] * np.log(x[f"{year}_gdp_d"]))
                    + (grav_model_coeff["distw_log"] * np.log(x["distw"]))
                    + (
                        grav_model_coeff["democracy_o"]
                        - (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_o"]
                    + (
                        grav_model_coeff["democracy_d"]
                        - (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_d"]
                ),
                axis=1,
            )
        bilateral_data.fillna(0, inplace=True)
        bilateral_data.to_csv(output_path / "dataframe" / f"bilateral_data_{c}_{r}.csv")
        return bilateral_data
    if c == 3:
        for year in YEARS:
            bilateral_data["trade_" + ship_type + str(year)] = bilateral_data.apply(
                lambda x: np.exp(
                    (grav_model_coeff["intercept"])
                    + (grav_model_coeff["pop_o_log"] * np.log(x[f"{year}_pop_o"]))
                    + (grav_model_coeff["pop_d_log"] * np.log(x[f"{year}_pop_d"]))
                    + (grav_model_coeff["gdp_o_log"] * np.log(x[f"{year}_gdp_o"]))
                    + (grav_model_coeff["gdp_d_log"] * np.log(x[f"{year}_gdp_d"]))
                    + (grav_model_coeff["distw_log"] * np.log(x["distw"]))
                    + (
                        grav_model_coeff["democracy_o"]
                        - (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_o"]
                    + (
                        grav_model_coeff["democracy_d"]
                        - (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_d"]
                ),
                axis=1,
            )
        bilateral_data.fillna(0, inplace=True)
        bilateral_data.to_csv(output_path / "dataframe" / f"bilateral_data_{c}_{r}.csv")
        return bilateral_data
    if c == 4:
        for year in YEARS:
            bilateral_data["trade_" + ship_type + str(year)] = bilateral_data.apply(
                lambda x: np.exp(
                    (grav_model_coeff["intercept"])
                    + (grav_model_coeff["pop_o_log"] * np.log(x[f"{year}_pop_o"]))
                    + (grav_model_coeff["pop_d_log"] * np.log(x[f"{year}_pop_d"]))
                    + (grav_model_coeff["gdp_o_log"] * np.log(x[f"{year}_gdp_o"]))
                    + (grav_model_coeff["gdp_d_log"] * np.log(x[f"{year}_gdp_d"]))
                    + (grav_model_coeff["distw_log"] * np.log(x["distw"]))
                    + (
                        grav_model_coeff["democracy_o"]
                        + (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_o"]
                    + (
                        grav_model_coeff["democracy_d"]
                        + (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                    )
                    * x[f"{year}_democracy_d"]
                ),
                axis=1,
            )
        bilateral_data.fillna(0, inplace=True)
        bilateral_data.to_csv(output_path / "dataframe" / f"bilateral_data_{c}_{r}.csv")
        return bilateral_data
    elif c == 5:
        for year in YEARS:
            bilateral_data["trade_" + ship_type + str(year)] = bilateral_data.apply(
                lambda x: np.exp(
                    (grav_model_coeff["intercept"])
                    + (grav_model_coeff["pop_o_log"] * np.log(x[f"{year}_pop_o"]))
                    + (grav_model_coeff["pop_d_log"] * np.log(x[f"{year}_pop_d"]))
                    + (grav_model_coeff["gdp_o_log"] * np.log(x[f"{year}_gdp_o"]))
                    + (grav_model_coeff["gdp_d_log"] * np.log(x[f"{year}_gdp_d"]))
                    + (grav_model_coeff["distw_log"] * np.log(x["distw"]))
                    + (
                        grav_model_coeff["democracy_o"]
                        + (abs(grav_model_coeff["democracy_o"]) * float(Scaling))
                        * x[f"{year}_democracy_o"]
                    )
                    + (
                        grav_model_coeff["democracy_d"]
                        + (abs(grav_model_coeff["democracy_d"]) * float(Scaling))
                        * x[f"{year}_democracy_d"]
                    )
                ),
                axis=1,
            )
        bilateral_data.fillna(0, inplace=True)
        bilateral_data.to_csv(output_path / "dataframe" / f"bilateral_data_{c}_{r}.csv")
        return bilateral_data


def merge_rivalry_clusters(rivalry, cluster):
    for r in rivalry:
        for c in cluster:
            print(c)
            print(r)
            globals()[f"bilateral_data_{c}_{r}"] = pd.read_csv(
                os.path.join(output_path, "dataframe", f"bilateral_data_{c}_{r}.csv"),
                low_memory=False,
                index_col=[0],
            )
            if c == 5:
                globals()[f"bilateral_data_{c}_{r}"]["cluster"] = "5"
            else:
                pass
            globals()[f"bilateral_data_{c}_{r}"].loc[
                ~(globals()[f"bilateral_data_{c}_{r}"] == int(0)).all(axis=1)
            ]
            globals()[f"bilateral_data_{c}_{r}"].loc[
                globals()[f"bilateral_data_{c}_{r}"]["ssp"] == int("3"), "ssp"
            ] = f"{r}"

    global bilateral_data_merged
    bilateral_data_merged = pd.DataFrame(globals()[f"bilateral_data_{c}_{r}"])
    for r in rivalry:
        for c in cluster:
            bilateral_data_merged = pd.concat(
                [bilateral_data_merged, globals()[f"bilateral_data_{c}_{r}"]]
            )
    bilateral_data_merged.to_csv(
        output_path / "dataframe" / "bilateral_data_merged_rivalry.csv"
    )


def main(
    rivalry,
    ship_types,
    clusters,
    clusters_adj,
    YEARS_HIST=np.arange(1996, 2018),
    output_path="outputs",
    solver: str = "gm",
    iterlimit: int = 1000000000,
    cluster: str = "False",
    threshold: str = "True",
) -> bool:
    """Run the entire analysis."""
    global DATA
    if redo_calc["bilateral_projection"] == True:
        if redo_calc["gm_estimation"] == True:
            read_data(YEARS_HIST)
            DATA.to_csv(output_path / "dataframe" / "DATA_b4_preproc.csv")
            if redo_calc["proc_all_data"] == True:
                # Merge data or import pre-processed dataframe
                if redo_calc["preproc_gm_data"] == True:
                    preproc_gm_data(ship_types, YEARS_HIST, output_path=output_path)
                else:
                    names_list = []
                    for ship in ship_types:
                        names_list.append("const" + ship + "_tw")
                    DATA = pd.read_csv(
                        os.path.join(
                            output_path, "dataframe", "CEPII_preproc.csv"
                        ),
                        low_memory=False,
                        index_col=[0],
                    )
                    for ship in ship_types:
                        dat2 = DATA.loc[(DATA["year"] == 2015)]
                        dat2 = dat2[[f"{ship}", "iso3_o", "iso3_d"]]
                        dat2[f"const_{ship}"] = dat2[f"{ship}"]
                        DATA = DATA.merge(
                            dat2[[f"const_{ship}", "iso3_o", "iso3_d"]],
                            how="inner",
                            on=["iso3_o", "iso3_d"],
                        )
                    DATA["gdp_o_log"] = np.log(DATA["gdp_o"])
                    DATA["gdp_d_log"] = np.log(DATA["gdp_d"])
                    DATA["gdpcap_ppp_d_log"] = np.log(DATA["gdpcap_ppp_d"])
                    DATA["gdpcap_ppp_o_log"] = np.log(DATA["gdpcap_ppp_o"])
                ssp_data(regressors_gm, output_path)
            else:
                DATA = pd.read_csv(
                    os.path.join(output_path, "dataframe", "intermediate_data.csv")
                )
            DATA.dropna()
            DATA.replace([np.inf, -np.inf], 1, inplace=True)
            DATA = DATA.drop_duplicates(keep=False)
            DATA.to_csv(output_path / "dataframe" / "DATA_after_const_and_cleaning.csv")
            if cluster == True:
                build_cluster()
            # Estimate OLS regression
            global COEF
            global p
            global SE
            for c in clusters:
                COEF.clear()
                for s in ship_types:
                    estimate_gm(s, c)
                globals()[f"COEF_{c}"] = copy.deepcopy(COEF)
                globals()[f"SE_{c}"] = copy.deepcopy(SE)
                globals()[f"p_{c}"] = copy.deepcopy(p)

            bilateral_data = exo_data_for_projection(YEARS, ship_types)
            bilateral_data.to_csv(
                output_path / "dataframe" / "bilateral_data_after_exo.csv"
            )
        else:  # if redo_calc['gm_estimation'] == False
            bilateral_data = pd.read_csv(
                os.path.join(output_path, "dataframe", "bilateral_data_test.csv")
            )
        bilateral_data_cluster()
        # Estimate projections and save
        for r in rivalry:
            for c in clusters:
                bilateral_data = globals()[f"bilateral_data_{c}"]
                for s in ship_types:
                    bilateral_data = globals()[f"bilateral_data_{c}"]
                    bilateral_data = run_gm_cluster(
                        r, s, c, bilateral_data, YEARS_HIST, output_path
                    )
                    globals()[f"bilateral_data_{c}_{r}"] = bilateral_data
        merge_rivalry_clusters(rivalry, clusters)
        bilateral_data = bilateral_data_merged
        bilateral_data.to_csv(
            os.path.join(output_path, "dataframe", "bilateral_data_cluster.csv")
        )
        return True


if __name__ == "__main__":
    print("start")
    for key in INDEP_VARS.keys():
        assert len(INDEP_VARS[key]) == len(
            set(INDEP_VARS[key])
        ), f"INDEP_VARS[{key}] contains duplicates"
    output_path = Path(__file__).resolve().parents[1].joinpath("outputs")
    main(
        ship_types=[
            "container_value",
            "liquified gas_value",
            "chemical_value",
            "refrigerated cargo_value",
            "ro-ro_value",
            "oil_value",
            "bulk dry_value",
        ],
        YEARS_HIST=list(np.arange(1995, 2018)),
        output_path=output_path,
        solver="gm",
        iterlimit=3,
        clusters=[1, 2, 3, 4, 5],
        clusters_adj=[1, 2, 3],
        cluster=True,
        threshold=False,
        rivalry=[1, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000],
    )
    print("----------\nSuccess!\n------")
