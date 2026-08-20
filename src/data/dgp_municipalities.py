"""
DGP de Covariáveis Municipais do Ceará.
---------------------------------------
Gera covariáveis socioeconômicas sintéticas (PIB per capita, população, IDH-M,
distância até Fortaleza, mesorregiões, receita tributária per capita, taxa de urbanização)
e malha GeoJSON simplificada para ~170 municípios do CE.

Os parâmetros do DGP foram calibrados com base em estatísticas descritivas públicas
do IBGE (PIB dos Municípios, Censo), PNUD (Atlas Brasil) e dados de infraestrutura.
"""

import json
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dados de municípios chave do Ceará com códigos IBGE oficiais de 7 dígitos, coordenadas e mesorregiões
CEARA_MUNICIPALITIES_DATA: List[Dict] = [
    {"code": 2304400, "name": "Fortaleza", "lat": -3.7319, "lon": -38.5267, "meso": "Metropolitana de Fortaleza", "idhm": 0.754, "pop": 2703391, "pib_pc": 26500.0, "senai": 1},
    {"code": 2307650, "name": "Maracanaú", "lat": -3.8767, "lon": -38.6256, "meso": "Metropolitana de Fortaleza", "idhm": 0.686, "pop": 229458, "pib_pc": 42100.0, "senai": 1},
    {"code": 2303709, "name": "Caucaia", "lat": -3.7361, "lon": -38.6531, "meso": "Metropolitana de Fortaleza", "idhm": 0.682, "pop": 365212, "pib_pc": 21800.0, "senai": 0},
    {"code": 2312908, "name": "Sobral", "lat": -3.6833, "lon": -40.3500, "meso": "Noroeste Cearense", "idhm": 0.714, "pop": 210711, "pib_pc": 24900.0, "senai": 1},
    {"code": 2307304, "name": "Juazeiro do Norte", "lat": -7.2139, "lon": -39.3156, "meso": "Sul Cearense", "idhm": 0.694, "pop": 276264, "pib_pc": 18200.0, "senai": 1},
    {"code": 2304202, "name": "Crato", "lat": -7.2339, "lon": -39.4086, "meso": "Sul Cearense", "idhm": 0.713, "pop": 133039, "pib_pc": 15400.0, "senai": 0},
    {"code": 2305506, "name": "Iguatu", "lat": -6.3589, "lon": -39.2989, "meso": "Centro-Sul Cearense", "idhm": 0.677, "pop": 103022, "pib_pc": 14900.0, "senai": 0},
    {"code": 2311306, "name": "Quixadá", "lat": -4.9714, "lon": -39.0153, "meso": "Sertões Cearenses", "idhm": 0.659, "pop": 88782, "pib_pc": 13800.0, "senai": 0},
    {"code": 2311801, "name": "Russas", "lat": -4.9403, "lon": -37.9758, "meso": "Jaguaribe", "idhm": 0.669, "pop": 79821, "pib_pc": 17200.0, "senai": 0},
    {"code": 2302800, "name": "Canindé", "lat": -4.3586, "lon": -39.3122, "meso": "Norte Cearense", "idhm": 0.612, "pop": 77484, "pib_pc": 11500.0, "senai": 0},
    {"code": 2304103, "name": "Crateús", "lat": -5.1783, "lon": -40.6775, "meso": "Sertões Cearenses", "idhm": 0.663, "pop": 76390, "pib_pc": 12800.0, "senai": 0},
    {"code": 2300200, "name": "Aquiraz", "lat": -3.9006, "lon": -38.3911, "meso": "Metropolitana de Fortaleza", "idhm": 0.672, "pop": 81077, "pib_pc": 31200.0, "senai": 0},
    {"code": 2305209, "name": "Horizonte", "lat": -4.1006, "lon": -38.4975, "meso": "Metropolitana de Fortaleza", "idhm": 0.658, "pop": 68884, "pib_pc": 38900.0, "senai": 1},
    {"code": 2309607, "name": "Pacajus", "lat": -4.1706, "lon": -38.4606, "meso": "Metropolitana de Fortaleza", "idhm": 0.650, "pop": 73188, "pib_pc": 28400.0, "senai": 0},
    {"code": 2313302, "name": "Tauá", "lat": -6.0022, "lon": -40.2936, "meso": "Sertões Cearenses", "idhm": 0.633, "pop": 60464, "pib_pc": 12100.0, "senai": 0},
    {"code": 2305001, "name": "Eusébio", "lat": -3.8886, "lon": -38.4522, "meso": "Metropolitana de Fortaleza", "idhm": 0.701, "pop": 54337, "pib_pc": 68500.0, "senai": 0},
    {"code": 2313401, "name": "Tianguá", "lat": -3.7317, "lon": -40.9919, "meso": "Noroeste Cearense", "idhm": 0.662, "pop": 76592, "pib_pc": 14100.0, "senai": 0},
    {"code": 2301000, "name": "Aracati", "lat": -4.5617, "lon": -37.7697, "meso": "Jaguaribe", "idhm": 0.655, "pop": 75110, "pib_pc": 16900.0, "senai": 0},
    {"code": 2302602, "name": "Camocim", "lat": -2.9017, "lon": -40.8419, "meso": "Noroeste Cearense", "idhm": 0.644, "pop": 64000, "pib_pc": 11800.0, "senai": 0},
    {"code": 2302107, "name": "Baturité", "lat": -4.3286, "lon": -38.8847, "meso": "Norte Cearense", "idhm": 0.647, "pop": 35848, "pib_pc": 13200.0, "senai": 0},
]

# Nomes complementares para cobrir a malha inteira de municípios do estado
SECONDARY_MUN_NAMES = [
    "Abaiara", "Acarape", "Acaraú", "Acopiara", "Alcântaras", "Altaneira", "Alto Santo", "Amontada",
    "Antonina do Norte", "Apuiarés", "Araripe", "Aratuba", "Arneiroz", "Assaré", "Aurora", "Baixio",
    "Banabuiú", "Barbalha", "Barreira", "Barro", "Barroquinha", "Bela Cruz", "Boa Viagem", "Brejo Santo",
    "Campos Sales", "Capistrano", "Caridade", "Cariré", "Caririaçu", "Cariús", "Carnaubal", "Cascavel",
    "Catarina", "Catunda", "Cedro", "Chaval", "Choró", "Chorozinho", "Coreaú", "Croatá",
    "Cruz", "Deputado Irapuan Pinheiro", "Ererê", "Farias Brito", "Forquilha", "Fortim", "Frecheirinha", "General Sampaio",
    "Graça", "Granja", "Granjeiro", "Groaíras", "Guaiúba", "Guaramiranga", "Hidrolândia", "Ibaretama",
    "Ibiapina", "Ibicuitinga", "Icapuí", "Ipaumirim", "Ipu", "Ipueiras", "Iracema", "Irauçuba",
    "Itaiçaba", "Itapajé", "Itapipoca", "Itapiúna", "Itarema", "Itatira", "Jaguaretama",
    "Jaguaribara", "Jaguaribe", "Jaguaruana", "Jardim", "Jati", "Jijoca de Jericoacoara", "Jucás", "Lavras da Mangabeira",
    "Limoeiro do Norte", "Madalena", "Maranguape", "Marco", "Martinópole", "Massapê", "Mauriti", "Meruoca",
    "Milagres", "Milhã", "Miraíma", "Missão Velha", "Mombaça", "Monsenhor Tabosa", "Morada Nova", "Moraújo",
    "Morrinhos", "Mucambo", "Mulungu", "Nova Olinda", "Nova Russas", "Novo Oriente", "Ocara", "Orós",
    "Pacoti", "Pacujá", "Palhano", "Palmácia", "Paracuru", "Paraipaba", "Parambu", "Paramoti",
    "Pedra Branca", "Penaforte", "Pentecoste", "Pereiro", "Pindoretama", "Piquet Carneiro", "Pires Ferreira", "Poranga",
    "Portelândia", "Potengi", "Potiretama", "Quixelô", "Quixeramobim", "Quixeré", "Redenção", "Reriutaba",
    "Salitre", "Santa Quitéria", "Santana do Acaraú", "Santana do Cariri", "São Benedito", "São Gonçalo do Amarante", "São Luís do Curu", "Senador Pompeu",
    "Senador Sá", "Solonópole", "Tabuleiro do Norte", "Tamboril", "Tarrafas", "Tejuçuoca", "Tururu",
    "Ubajara", "Umari", "Umirim", "Uruburetama", "Uruoca", "Varjota", "Várzea Alegre", "Viçosa do Ceará"
]


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float = -3.7319, lon2: float = -38.5267) -> float:
    """Calcula a distância Haversine em km de uma coordenada até a capital Fortaleza (-3.7319, -38.5267)."""
    R = 6371.0  # Raio da Terra em km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return float(R * c)


def load_ibge_covariates(params_path: str = "config/params.yaml") -> pd.DataFrame:
    """
    Gera covariáveis socioeconômicas sintéticas calibradas para os municípios do Ceará.

    Retorna:
        pd.DataFrame com código municipal, nome, PIB per capita, IDH-M, população,
        distância até Fortaleza, mesorregião, presença de unidade técnica, receita tributária e urbanização.
    """
    np.random.seed(42)
    records = list(CEARA_MUNICIPALITIES_DATA)

    meso_list = [
        "Metropolitana de Fortaleza", "Noroeste Cearense", "Norte Cearense",
        "Sertões Cearenses", "Jaguaribe", "Centro-Sul Cearense", "Sul Cearense"
    ]

    existing_names = {r["name"] for r in records}
    code_counter = 2300050

    for name in SECONDARY_MUN_NAMES:
        if name in existing_names or len(records) >= 170:
            continue
        code_counter += 10
        # Distribuição espacial dentro da caixa delimitadora do Ceará
        lat = float(np.random.uniform(-7.4, -2.9))
        lon = float(np.random.uniform(-41.2, -37.7))
        meso = np.random.choice(meso_list, p=[0.15, 0.20, 0.15, 0.20, 0.10, 0.10, 0.10])

        # Correlações socioeconômicas realistas
        idhm = float(np.clip(np.random.normal(0.64, 0.04), 0.52, 0.76))
        pop = int(np.random.lognormal(mean=9.3, sigma=0.8))
        pib_pc = float(np.clip(idhm * 28000 + np.random.normal(0, 3000), 7500, 55000))
        senai = 1 if (pib_pc > 30000 and np.random.rand() > 0.6) else 0

        records.append({
            "code": code_counter,
            "name": name,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "meso": meso,
            "idhm": round(idhm, 3),
            "pop": pop,
            "pib_pc": round(pib_pc, 2),
            "senai": senai
        })
        existing_names.add(name)

    df = pd.DataFrame(records)

    # Calcular distância geográfica até Fortaleza
    df["dist_capital_km"] = df.apply(
        lambda row: calculate_haversine_distance(row["lat"], row["lon"]), axis=1
    ).round(2)

    # Variáveis derivadas
    df["tax_revenue_per_capita"] = (df["pib_pc"] * np.random.uniform(0.04, 0.09, len(df))).round(2)
    df["urbanization_rate"] = np.clip(
        0.45 + 0.35 * (df["idhm"] - 0.5) / 0.25 + np.random.normal(0, 0.05, len(df)), 0.30, 0.98
    ).round(3)

    # Renomear colunas para o esquema padrão do projeto
    df = df.rename(columns={
        "code": "mun_code",
        "name": "mun_name",
        "meso": "mesoregion",
        "pib_pc": "pib_per_capita",
        "pop": "population",
        "senai": "senai_presence"
    })

    logger.info(f"Covariáveis municipais geradas para {len(df)} municípios do Ceará.")
    return df


def generate_ceara_geojson(df_ibge: pd.DataFrame, output_path: str = "data/geojson/ceara_municipalities.json") -> Dict:
    """
    Gera polígonos GeoJSON simplificados para os municípios do Ceará.
    Salva o arquivo em output_path.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for _, row in df_ibge.iterrows():
        lat, lon = row["lat"], row["lon"]
        delta = 0.04
        coords = [[
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta]
        ]]
        feature = {
            "type": "Feature",
            "id": str(row["mun_code"]),
            "properties": {
                "name": row["mun_name"],
                "code": row["mun_code"],
                "mesoregion": row["mesoregion"]
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    logger.info(f"GeoJSON do Ceará salvo com {len(features)} municípios em {output_path}")
    return geojson_data


if __name__ == "__main__":
    df_covs = load_ibge_covariates()
    generate_ceara_geojson(df_covs)
    print(df_covs.head())
