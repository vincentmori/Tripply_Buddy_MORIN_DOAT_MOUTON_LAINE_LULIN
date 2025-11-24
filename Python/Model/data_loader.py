# data_loader.py
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_season(date):
    try:
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    except:
        return 'Autumn'


def load_and_process_data(csv_path):
    print(f"--- [Data Engineer] Chargement de {csv_path} ---")
    df = pd.read_csv(csv_path)
    df.drop(columns=["Local Transport Options"],inplace=True)
    print(df.columns)
    mappings = {}
    # Mappings
    user_names = df['Traveler name'].unique()
    mappings['User'] = {name: i for i, name in enumerate(user_names)}
    df['uid'] = df['Traveler name'].map(mappings['User'])

    dest_names = df['Destination'].unique()
    mappings['Destination'] = {name: i for i, name in enumerate(dest_names)}
    df['dest_id'] = df['Destination'].map(mappings['Destination'])

    cat_cols = ['Accommodation type', 'Transportation type', 'Traveler nationality', 'Traveler gender']
    for col in cat_cols:
        unique_vals = df[col].unique()
        mappings[col] = {name: i for i, name in enumerate(unique_vals)}

    df['acc_id'] = df['Accommodation type'].map(mappings['Accommodation type'])
    df['trans_id'] = df['Transportation type'].map(mappings['Transportation type'])
    df['nationality_id'] = df['Traveler nationality'].map(mappings['Traveler nationality'])
    df['gender_id'] = df['Traveler gender'].map(mappings['Traveler gender'])

    df['Start date'] = pd.to_datetime(df['Start date'])
    df['season'] = df['Start date'].apply(get_season)
    mappings['season'] = {name: i for i, name in enumerate(df['season'].unique())}
    df['season_id'] = df['season'].map(mappings['season'])

    num_cols = ['Accommodation cost', 'Transportation cost', 'Duration (days)', 'Traveler age']
    for col in num_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
        df[col] = df[col].astype('float32')

    return df, mappings


def build_graph(df):
    print("--- [Data Engineer] Construction du Graphe ---")
    user_df = df[['uid', 'gender_id', 'nationality_id', 'Traveler age']].drop_duplicates('uid').sort_values('uid')
    scaler_age = StandardScaler()
    age_scaled = scaler_age.fit_transform(user_df[['Traveler age']].values)

    user_x = torch.cat([
        torch.tensor(user_df['gender_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(user_df['nationality_id'].values, dtype=torch.float).unsqueeze(1),
        torch.tensor(age_scaled, dtype=torch.float)
    ], dim=1)

    dest_df = df[['dest_id']].drop_duplicates('dest_id').sort_values('dest_id')
    dest_x = torch.ones((len(dest_df), 1), dtype=torch.float)

    src = torch.tensor(df['uid'].values, dtype=torch.long)
    dst = torch.tensor(df['dest_id'].values, dtype=torch.long)

    edge_attr_cat = torch.stack([
        torch.tensor(df['acc_id'].values, dtype=torch.long),
        torch.tensor(df['trans_id'].values, dtype=torch.long),
        torch.tensor(df['season_id'].values, dtype=torch.long)
    ], dim=1)

    scaler_edges = StandardScaler()
    edge_nums = df[['Accommodation cost', 'Transportation cost', 'Duration (days)']].values
    edge_attr_num = torch.tensor(scaler_edges.fit_transform(edge_nums), dtype=torch.float)

    data = HeteroData()
    data['user'].num_nodes = len(user_df)
    data['user'].x = user_x
    data['destination'].num_nodes = len(dest_df)
    data['destination'].x = dest_x
    data['user', 'visits', 'destination'].edge_index = torch.stack([src, dst], dim=0)
    data['user', 'visits', 'destination'].edge_attr_cat = edge_attr_cat
    data['user', 'visits', 'destination'].edge_attr_num = edge_attr_num
    data['destination', 'rev_visits', 'user'].edge_index = torch.stack([dst, src], dim=0)

    return data