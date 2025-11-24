# train.py
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
import os
import json
import pickle

# Imports locaux (Modularité)
from model import HGIB_Context_Model
from data_loader import load_and_process_data, build_graph

# ==========================================
# HYPERPARAMÈTRES (MLE-Ops Responsibility)
# ==========================================
CSV_FILE = 'synthetic_travel_data_daily_cost_coherent.csv'
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
LEARNING_RATE = 0.001
EPOCHS = 500  # On peut augmenter à 100
BETA = 0  # Poids du Information Bottleneck (KL Divergence)
ARTIFACTS_DIR = "artifacts"  # Dossier où on sauvegarde le modèle


def train_one_epoch(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    pred, mu, logstd = model(train_data)
    target = train_data['user', 'visits', 'destination'].edge_label

    # 1. Reconstruction Loss (BCE)
    recons_loss = F.binary_cross_entropy_with_logits(pred, target)

    # 2. KL Divergence (Bottleneck)
    kl_loss = 0
    for key in mu.keys():
        kl_loss += -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd[key] - mu[key] ** 2 - logstd[key].exp() ** 2, dim=1)
        )

    # Loss Totale
    loss = recons_loss + (BETA * kl_loss)

    loss.backward()
    optimizer.step()

    return loss.item(), recons_loss.item()


@torch.no_grad()
def evaluate(model, val_data):
    """Validation simple pour vérifier que la loss descend aussi sur les données inconnues"""
    model.eval()
    pred, mu, logstd = model(val_data)
    target = val_data['user', 'visits', 'destination'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss.item()


def main():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- [MLE-Ops] Training Start on {device} ---")

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    # 2. Chargement des données (Appel au Data Engineer / PL)
    if not os.path.exists(CSV_FILE):
        print(f"ERREUR: Fichier {CSV_FILE} introuvable.")
        return

    df, mappings = load_and_process_data(CSV_FILE)
    data = build_graph(df)

    # 3. Split Train/Val/Test
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[('user', 'visits', 'destination')],
        rev_edge_types=[('destination', 'rev_visits', 'user')],
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)

    # Transfert sur GPU si dispo
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # 4. Initialisation du Modèle (Appel au MLE-Core)
    num_acc = len(mappings['Accommodation type'])
    num_trans = len(mappings['Transportation type'])
    num_season = len(mappings['season'])
    # --- AJOUTS ICI ---
    num_users = data['user'].num_nodes
    num_dests = data['destination'].num_nodes

    model = HGIB_Context_Model(
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        metadata=data.metadata(),
        num_acc=num_acc,
        num_trans=num_trans,
        num_season=num_season,
        num_users=num_users,
        num_dests=num_dests
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Boucle d'Entraînement
    print(f"--- Début de l'entraînement pour {EPOCHS} époques ---")

    for epoch in range(1, EPOCHS + 1):
        train_loss, recons_loss = train_one_epoch(model, train_data, optimizer)
        val_loss = evaluate(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("--- Entraînement terminé ---")

    # 6. Sauvegarde des Artefacts (CRUCIAL pour la semaine 3 et l'API)
    print("--- Sauvegarde du modèle et des mappings ---")

    # Sauvegarde des poids du modèle
    model_path = os.path.join(ARTIFACTS_DIR, "hgib_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé : {model_path}")

    # Sauvegarde des mappings (nécessaire pour l'API pour traduire les ID)
    # On ne peut pas sauvegarder tout le dataframe, juste les dictionnaires
    mappings_path = os.path.join(ARTIFACTS_DIR, "mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"Mappings sauvegardés : {mappings_path}")

    # Sauvegarde des dimensions (pour réinitialiser le modèle dans l'API)
    config = {
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "num_acc": num_acc,
        "num_trans": num_trans,
        "num_season": num_season
    }
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Config sauvegardée : {config_path}")


if __name__ == "__main__":
    main()