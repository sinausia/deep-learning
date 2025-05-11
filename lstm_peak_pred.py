import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os
from datetime import datetime
import matplotlib as mpl

mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none' 

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = [
    {'path': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Li'},
    {'path': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Na'},
    {'path': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'K'},
    {'path': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/1101_to_3999/Interfacial_layer_withoutbackground_correction_integrated_areas.csv', 'cation': 'Cs'},
]

def load_integrated_data(file_path):
    try:
        # Load with header
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        print(f"Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def prepare_data_for_lstm_with_cation(X, y, cation_encoding, time_steps=5):
    """
    Prepare data for LSTM by creating sequences of time_steps length
    X: input features (all peaks except the target peak)
    y: target values (the peak being predicted)
    cation_encoding: one-hot encoded cation identity
    time_steps: number of time steps to include in each sequence
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        seq = X[i:i+time_steps].copy()
        X_seq.append(seq)
        y_seq.append(y[i+time_steps])

    X_seq_np = np.array(X_seq)
    y_seq_np = np.array(y_seq)

    cation_repeated = np.tile(cation_encoding, (len(X_seq_np), 1))
    
    return X_seq_np, y_seq_np, cation_repeated

class CationLSTMModel(nn.Module):
    def __init__(self, input_size, cation_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.3):
        super(CationLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size2 + cation_size, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate/2)  # Less dropout on final layers
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x, cation):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        lstm_out = lstm2_out[:, -1, :]
        combined = torch.cat((lstm_out, cation), dim=1)

        x = self.relu(self.fc1(combined))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

#correlation matrix
def plot_correlation_matrix(df, title, output_dir=None):
    plt.figure(figsize=(14, 12), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette("Blues", as_cmap=True)
    ax = sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap=cmap, 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 10},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    plt.title(title, fontsize=18, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()


def plot_feature_importance(importance, feature_names, title, output_dir=None):
    plt.figure(figsize=(14, 10), dpi=300)
    
    # Sort indices and data
    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(sorted_idx)))
    bars = plt.barh(range(len(sorted_idx)), sorted_importance, color=colors)
    plt.yticks(range(len(sorted_idx)), sorted_names, fontsize=14)
    plt.xlabel('Feature Importance', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01*max(importance), i, f"{v:.2e}", va='center', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()

# predictions vs actual
def plot_predictions(y_true, y_pred, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    scatter = plt.scatter(y_true, y_pred, c=y_true, cmap='Blues', 
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Actual Values', fontsize=14)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lin
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    textstr = f'$R^2 = {r2:.4f}$\n$MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, bbox=props, verticalalignment='top')    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()
    return r2, mse

def plot_cation_predictions(y_true, y_pred, cations, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)
    cation_colors = {'Na': '#1f77b4', 'K': '#2ca02c', 'Cs': '#d62728', 'Li': '#9467bd'}
    for cation in set(cations):
        mask = np.array(cations) == cation
        if sum(mask) > 0:  # Only if we have data for this cation
            plt.scatter(
                y_true[mask], 
                y_pred[mask], 
                color=cation_colors.get(cation, 'gray'),
                label=f"{cation} (R²={r2_score(y_true[mask], y_pred[mask]):.4f})",
                alpha=0.7,
                s=80,
                edgecolor='white',
                linewidth=0.5
            )
    
    # Add diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
  
    overall_r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    textstr = f'Overall $R^2 = {overall_r2:.4f}$\nOverall $MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=14, bbox=props, verticalalignment='top')
    
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')
    
    plt.close()
    return overall_r2, mse


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device=device, verbose=True):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, cations, targets in train_loader:
            inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, cations)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, cations, targets in val_loader:
                inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
                outputs = model(inputs, cations)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Main function to analyze with leave-one-out approach
def analyze_leave_one_out():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'/Users/danielsinausia/Documents/LSTM_leave_one_out_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    all_peak_data = []
    all_cations = []
    all_experiment_ids = []
    valid_peak_names = []  # Will store the intersection of available peaks across all experiments
    
    # load data and determine which peaks are available in all experiments
    for i, exp in enumerate(experiment_data):
        data = load_integrated_data(exp['path'])
        if data is None:
            continue

        peak_columns = [col for col in data.columns if col.startswith('Mean')]

        if i == 0:
            valid_peak_names = peak_columns
        else:
            valid_peak_names = [peak for peak in valid_peak_names if peak in peak_columns]
        
        all_peak_data.append(data[peak_columns].values)
        all_cations.extend([exp['cation']] * len(data))
        all_experiment_ids.extend([i] * len(data))

    print(f"Using {len(valid_peak_names)} peaks that are common across all experiments:")
    print(valid_peak_names)
    results = []
    
    # One-hot encode the cation information
    cation_encoder = OneHotEncoder(sparse_output=False)
    cation_encoded = cation_encoder.fit_transform(np.array(all_cations).reshape(-1, 1))

    combined_peaks = np.vstack([data[:, [valid_peak_names.index(peak) if peak in valid_peak_names else -1 
                                        for peak in valid_peak_names]] 
                               for data in all_peak_data])

    combined_df = pd.DataFrame(combined_peaks, columns=valid_peak_names)
    combined_df['Cation'] = all_cations
    plot_correlation_matrix(
        combined_df.drop('Cation', axis=1), 
        "Correlation Matrix of All FTIR Peaks", 
        output_dir
    )

    time_steps = 5
    epochs = 100
    cation_size = cation_encoded.shape[1]
    
    for target_idx, target_peak in enumerate(valid_peak_names):
        print(f"\n{'='*80}\nTraining model for target peak: {target_peak} ({target_idx+1}/{len(valid_peak_names)})\n{'='*80}")
        peak_dir = os.path.join(output_dir, f"peak_{target_peak.replace(' ', '_')}")
        os.makedirs(peak_dir, exist_ok=True)
        feature_peaks = [peak for peak in valid_peak_names if peak != target_peak]
        X_indices = [valid_peak_names.index(peak) for peak in feature_peaks]
        y_index = valid_peak_names.index(target_peak)
        
        X = combined_peaks[:, X_indices]
        y = combined_peaks[:, y_index]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        all_X_seq = []
        all_y_seq = []
        all_cation_seq = []
        all_cation_labels = []
        for exp_id in sorted(set(all_experiment_ids)):
            exp_indices = [i for i, eid in enumerate(all_experiment_ids) if eid == exp_id]

            exp_X = X_scaled[exp_indices]
            exp_y = y_scaled[exp_indices]
            exp_cation = cation_encoded[exp_indices[0]]  # All same cation in experiment
            exp_cation_label = all_cations[exp_indices[0]]
            
            # Create sequences
            if len(exp_indices) > time_steps:  # Only if we have enough data points
                X_seq, y_seq, _ = prepare_data_for_lstm_with_cation(exp_X, exp_y, exp_cation, time_steps)
                
                # Store
                all_X_seq.append(X_seq)
                all_y_seq.append(y_seq)
                all_cation_seq.extend([exp_cation] * len(X_seq))
                all_cation_labels.extend([exp_cation_label] * len(X_seq))

        X_seq = np.vstack(all_X_seq)
        y_seq = np.concatenate(all_y_seq)
        cation_seq = np.array(all_cation_seq)
        
        print(f"Prepared {len(X_seq)} sequences with {X_seq.shape[2]} features each")
        
        X_train, X_test, y_train, y_test, cation_train, cation_test, cation_labels_train, cation_labels_test = train_test_split(
            X_seq, y_seq, cation_seq, all_cation_labels, test_size=0.2, random_state=42, stratify=all_cation_labels
        )
        

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        cation_train_tensor = torch.FloatTensor(cation_train)
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        cation_test_tensor = torch.FloatTensor(cation_test)
        
        train_size = int(0.8 * len(X_train_tensor))
        val_size = len(X_train_tensor) - train_size
        
        train_dataset = TensorDataset(
            X_train_tensor[:train_size], 
            cation_train_tensor[:train_size], 
            y_train_tensor[:train_size]
        )
        val_dataset = TensorDataset(
            X_train_tensor[train_size:], 
            cation_train_tensor[train_size:], 
            y_train_tensor[train_size:]
        )
        test_dataset = TensorDataset(
            X_test_tensor, 
            cation_test_tensor, 
            y_test_tensor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)

        input_size = X_train.shape[2]  # Number of features
        model = CationLSTMModel(input_size, cation_size, dropout_rate=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print(f"Training LSTM model for {target_peak}...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device
        )
        
        # Plot training history
        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(train_losses, color='#9467bd', linewidth=2.5, label='Training Loss')
        plt.plot(val_losses, color='#e377c2', linewidth=2.5, label='Validation Loss')
        
        plt.fill_between(range(len(train_losses)), train_losses, alpha=0.2, color='#9467bd')
        plt.fill_between(range(len(val_losses)), val_losses, alpha=0.2, color='#e377c2')
        
        plt.title(f'Training History: {target_peak}', fontsize=18, pad=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(peak_dir, 'Training_History.png'), bbox_inches='tight')
        plt.savefig(os.path.join(peak_dir, 'Training_History.svg'), bbox_inches='tight', format='svg')
        plt.close()
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            all_y_pred = []
            all_y_true = []
            
            for inputs, cations, targets in test_loader:
                inputs, cations, targets = inputs.to(device), cations.to(device), targets.to(device)
                outputs = model(inputs, cations)
                
                all_y_pred.append(outputs.cpu().numpy())
                all_y_true.append(targets.cpu().numpy())
            
            y_pred = np.vstack(all_y_pred).flatten()
            y_true = np.vstack(all_y_true).flatten()

        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_original = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

        r2, mse = plot_predictions(
            y_true_original, 
            y_pred_original, 
            f"Predictions for {target_peak}", 
            peak_dir
        )

        cation_r2, cation_mse = plot_cation_predictions(
            y_true_original, 
            y_pred_original, 
            cation_labels_test,
            f"Predictions by Cation: {target_peak}",
            peak_dir
        )
        
        print(f"Calculating feature importance for {target_peak}...")
        
        feature_importance = np.zeros(len(feature_peaks))
        baseline_mse = mean_squared_error(y_true, y_pred)
        
        # Calculate feature importance by permuting each feature
        for feature_idx, feature_name in enumerate(feature_peaks):
            X_permuted = X_test.copy()
            # Permute the feature across all time steps
            for t in range(time_steps):
                orig_values = X_permuted[:, t, feature_idx].copy()
                # Permute the values
                np.random.shuffle(X_permuted[:, t, feature_idx])
                # Convert to tensor and predict
                X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
                
                with torch.no_grad():
                    y_permuted_pred = model(X_permuted_tensor, cation_test_tensor.to(device))
                    y_permuted_pred = y_permuted_pred.cpu().numpy().flatten()
                
                permuted_mse = mean_squared_error(y_true, y_permuted_pred)                
                feature_importance[feature_idx] += (permuted_mse - baseline_mse)
                X_permuted[:, t, feature_idx] = orig_values

        feature_importance /= time_steps
        
        # Plot feature importance
        plot_feature_importance(
            feature_importance, 
            feature_peaks, 
            f"Feature Importance for {target_peak}",
            peak_dir
        )
        
        # Identify top features
        top_n = min(5, len(feature_peaks))
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_peaks[i] for i in top_indices]
        
        # Save model
        torch.save(model.state_dict(), os.path.join(peak_dir, f'LSTM_model_{target_peak.replace(" ", "_")}.pt'))

        feature_importance_df = pd.DataFrame({
            'Feature': feature_peaks,
            'Importance': feature_importance
        })
        feature_importance_df.to_csv(os.path.join(peak_dir, 'feature_importance.csv'), index=False)
        
        # Save predictions and actual values
        results_df = pd.DataFrame({
            'Actual': y_true_original,
            'Predicted': y_pred_original,
            'Error': y_true_original - y_pred_original,
            'Cation': cation_labels_test
        })
        results_df.to_csv(os.path.join(peak_dir, 'prediction_results.csv'), index=False)

        cation_r2_dict = {}
        for cation in set(cation_labels_test):
            mask = np.array(cation_labels_test) == cation
            if sum(mask) > 0:
                cation_r2_dict[cation] = r2_score(y_true_original[mask], y_pred_original[mask])
        
        # summary statistics
        with open(os.path.join(peak_dir, 'summary_statistics.txt'), 'w') as f:
            f.write(f"Model for predicting '{target_peak}' using other peaks\n")
            f.write(f"Overall MSE: {mse:.4f}\n")
            f.write(f"Overall R²: {r2:.4f}\n")
            f.write("\nCation-Specific R² Values:\n")
            for cation, r2_val in cation_r2_dict.items():
                f.write(f"{cation}: {r2_val:.4f}\n")
            
            f.write(f"\nTop {top_n} important features:\n")
            for i, feature in enumerate(top_features):
                idx = feature_peaks.index(feature)
                f.write(f"{i+1}. {feature}: {feature_importance[idx]:.6f}\n")
        
        results.append({
            'target_peak': target_peak,
            'r2': r2,
            'mse': mse,
            'cation_r2': cation_r2_dict,
            'top_features': top_features,
            'feature_importance': {feat: imp for feat, imp in zip(feature_peaks, feature_importance)}
        })

    summary_df = pd.DataFrame({
        'Peak': [r['target_peak'] for r in results],
        'R²': [r['r2'] for r in results],
        'MSE': [r['mse'] for r in results]
    })
    
    # Sort by R²
    summary_df = summary_df.sort_values('R²', ascending=False)

    summary_df.to_csv(os.path.join(output_dir, 'peak_prediction_summary.csv'), index=False)
    plt.figure(figsize=(15, 10), dpi=300)
    bars = plt.barh(summary_df['Peak'], summary_df['R²'])
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(summary_df)))
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
    
    plt.xlabel('R² Score', fontsize=16)
    plt.ylabel('FTIR Peak', fontsize=16)
    plt.title('Prediction Performance for Each FTIR Peak', fontsize=18, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 1.0)
    for i, v in enumerate(summary_df['R²']):
        plt.text(v + 0.02, i, f"{v:.4f}", va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Peak_Prediction_Performance.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Peak_Prediction_Performance.svg'), bbox_inches='tight', format='svg')
    plt.close()
    

    relationship_matrix = np.zeros((len(valid_peak_names), len(valid_peak_names)))

    for result in results:
        target_idx = valid_peak_names.index(result['target_peak'])
        for feature, importance in result['feature_importance'].items():
            feature_idx = valid_peak_names.index(feature)
            relationship_matrix[target_idx, feature_idx] = importance
    relationship_df = pd.DataFrame(relationship_matrix, 
                                index=valid_peak_names, 
                                columns=valid_peak_names)
    
    # relationship heatmap
    plt.figure(figsize=(16, 14), dpi=300)
    sns.heatmap(relationship_df, cmap='Blues', annot=False, 
               xticklabels=True, yticklabels=True)
    plt.title('Feature Importance Relationships Between FTIR Peaks', fontsize=18, pad=20)
    plt.xlabel('Predictor Peak', fontsize=16)
    plt.ylabel('Target Peak', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Peak_Relationship_Heatmap.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Peak_Relationship_Heatmap.svg'), bbox_inches='tight', format='svg')
    plt.close()
    
    # network visualization of top relationships. For each peak, identify the top 3 predictors
    network_data = []
    for result in results:
        target = result['target_peak']
        top_3_features = []
        if result['top_features']:
            # Sort by importance and take top 3 or fewer
            top_3_features = sorted(
                [(feat, result['feature_importance'][feat]) for feat in result['top_features']],
                key=lambda x: x[1],
                reverse=True
            )[:min(3, len(result['top_features']))]
        
        for feature, importance in top_3_features:
            if importance > 0:  # Only include positive relationships
                network_data.append({
                    'source': feature,
                    'target': target,
                    'weight': importance
                })

  #network for visualization
    network_df = pd.DataFrame(network_data)
    network_df.to_csv(os.path.join(output_dir, 'peak_network_data.csv'), index=False)
#summary report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("# FTIR Peak Prediction Analysis Summary\n\n")
        
        f.write("## Overall Performance\n")
        f.write("Peaks sorted by prediction accuracy (R²):\n\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Peak']}: R² = {row['R²']:.4f}, MSE = {row['MSE']:.4f}\n")
        
        f.write("\n## Top Predictive Relationships\n")
        f.write("For each peak, the top 3 most important predictor peaks:\n\n")
        
        for result in results:
            target = result['target_peak']
            f.write(f"{target}:\n")
            
            # Sort by importance
            top_features = sorted(
                [(feat, result['feature_importance'][feat]) for feat in result['feature_importance']],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for i, (feature, importance) in enumerate(top_features):
                f.write(f"  {i+1}. {feature} (importance: {importance:.6f})\n")
            
            f.write("\n")
        
        f.write("\n## Cation-Specific Performance\n")
        for result in results:
            target = result['target_peak']
            f.write(f"{target}:\n")
            
            for cation, r2_val in result['cation_r2'].items():
                f.write(f"  {cation}: R² = {r2_val:.4f}\n")
            
            f.write("\n")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return summary_df, results

if __name__ == "__main__":
    print("Starting leave-one-out analysis for all FTIR peaks...")
    summary, results = analyze_leave_one_out()
    print("Analysis complete!")
