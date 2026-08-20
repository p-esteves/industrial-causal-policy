"""
Definição e Visualização do Grafo Causal (DAG).
-----------------------------------------------
Constrói o Grafo Dirigido Acíclico (DAG) via DoWhy e graphviz para formalizar
as premissas de identificação: unconfoundedness, positividade e SUTVA.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import dowhy
from dowhy import CausalModel
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_dag_gml_string() -> str:
    """Retorna a representação textual em formato GML do DAG do projeto."""
    gml = """
    graph [
        directed 1
        node [ id "fdi_incentive" label "fdi_incentive" ]
        node [ id "net_job_gain" label "net_job_gain" ]
        node [ id "pib_per_capita" label "pib_per_capita" ]
        node [ id "idhm" label "idhm" ]
        node [ id "population" label "population" ]
        node [ id "dist_capital_km" label "dist_capital_km" ]
        node [ id "senai_presence" label "senai_presence" ]
        node [ id "ind_emp_share" label "ind_emp_share" ]
        node [ id "n_establishments" label "n_establishments" ]
        node [ id "avg_industrial_wage" label "avg_industrial_wage" ]
        node [ id "urbanization_rate" label "urbanization_rate" ]
        node [ id "tax_revenue_per_capita" label "tax_revenue_per_capita" ]
        node [ id "unobserved_governance" label "unobserved_governance" ]

        edge [ source "pib_per_capita" target "fdi_incentive" ]
        edge [ source "pib_per_capita" target "net_job_gain" ]
        edge [ source "idhm" target "fdi_incentive" ]
        edge [ source "idhm" target "net_job_gain" ]
        edge [ source "dist_capital_km" target "fdi_incentive" ]
        edge [ source "dist_capital_km" target "net_job_gain" ]
        edge [ source "senai_presence" target "fdi_incentive" ]
        edge [ source "senai_presence" target "net_job_gain" ]
        edge [ source "ind_emp_share" target "fdi_incentive" ]
        edge [ source "ind_emp_share" target "net_job_gain" ]
        edge [ source "n_establishments" target "net_job_gain" ]
        edge [ source "avg_industrial_wage" target "net_job_gain" ]
        edge [ source "urbanization_rate" target "fdi_incentive" ]
        edge [ source "urbanization_rate" target "net_job_gain" ]
        edge [ source "tax_revenue_per_capita" target "fdi_incentive" ]
        edge [ source "tax_revenue_per_capita" target "net_job_gain" ]
        edge [ source "unobserved_governance" target "fdi_incentive" ]
        edge [ source "unobserved_governance" target "net_job_gain" ]
        edge [ source "fdi_incentive" target "net_job_gain" ]
    ]
    """
    return gml


COVARIATES = [
    "pib_per_capita", "idhm", "population", "dist_capital_km",
    "senai_presence", "ind_emp_share", "n_establishments",
    "avg_industrial_wage", "urbanization_rate", "tax_revenue_per_capita"
]


def build_dowhy_causal_model(df: pd.DataFrame) -> CausalModel:
    """
    Constrói o objeto CausalModel do DoWhy a partir das covariáveis e tratamento.
    
    Argumentos:
        df: DataFrame analítico.
        
    Retorna:
        Objeto dowhy.CausalModel.
    """
    model = CausalModel(
        data=df,
        treatment="fdi_incentive",
        outcome="net_job_gain",
        common_causes=COVARIATES
    )
    logger.info("DoWhy CausalModel construído com sucesso usando as covariáveis do projeto.")
    return model


def export_dag_visualization(output_path: str = "results/figures/causal_dag.png") -> None:
    """Salva a representação gráfica do DAG em output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dot_path = Path(output_path).with_suffix(".dot")
    dot_content = """digraph IndustrialCausalPolicy {
        rankdir=LR;
        node [shape=ellipse, style=filled, fillcolor="#1f2937", fontcolor="#f9fafb", fontname="Calibri"];
        edge [color="#6b7280", arrowhead=normal];

        subgraph cluster_confounders {
            label="Covariáveis (Confounders X)";
            color="#374151";
            fontcolor="#e5e7eb";
            "PIB pc"; "IDH-M"; "Dist. Capital"; "Capacitação Técnica"; "Ind. Share"; "Urbanização"; "Receita Trib.";
        }

        "FDI Incentive (T)" [fillcolor="#0284c7", fontcolor="#ffffff", shape=box, style="filled,bold"];
        "Net Job Gain (Y)" [fillcolor="#16a34a", fontcolor="#ffffff", shape=box, style="filled,bold"];
        "Governança Local (U)" [fillcolor="#dc2626", fontcolor="#ffffff", style="dashed,filled"];

        "PIB pc" -> "FDI Incentive (T)";
        "PIB pc" -> "Net Job Gain (Y)";
        "IDH-M" -> "FDI Incentive (T)";
        "IDH-M" -> "Net Job Gain (Y)";
        "Dist. Capital" -> "FDI Incentive (T)";
        "Dist. Capital" -> "Net Job Gain (Y)";
        "Capacitação Técnica" -> "FDI Incentive (T)";
        "Capacitação Técnica" -> "Net Job Gain (Y)";
        "Ind. Share" -> "FDI Incentive (T)";
        "Ind. Share" -> "Net Job Gain (Y)";
        "Governança Local (U)" -> "FDI Incentive (T)" [style=dashed];
        "Governança Local (U)" -> "Net Job Gain (Y)" [style=dashed];
        "FDI Incentive (T)" -> "Net Job Gain (Y)" [color="#0284c7", penwidth=2.5];
    }"""
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    logger.info(f"Especificação DOT do DAG salva em {dot_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/analytical_dataset.csv")
    model = build_dowhy_causal_model(df)
    export_dag_visualization()
