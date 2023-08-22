import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import tqdm
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
plt.style.use('ggplot')

def translate_features(feature_list):
    columns_german = {
    'Gross Margin': 'Rohertragsmarge', 'Current Ratio': 'Liquidität 3. Grades', 'Long-term Debt / Capital': 'Langfristige Schulden/Kapital', 'Debt/Equity Ratio': 'Verschuldungsgrad',
    'EBIT Margin': 'EBIT-Marge', 'Asset Turnover': 'Kapitalumschlag', 'ROE - Return On Equity': 'Eigenkapitalrendite', 'Return On Tangible Equity': 'RoTE',
        'Operating Cash Flow Per Share': 'Operativer Cashflow pro Aktie', 'Free Cash Flow Per Share': 'Freier Cashflow je Aktie', 'Sector': 'Industriesektor', 'ESG Rating': 'ESG',
        'ESG to Sector Average': 'ESG-Score zu Sektordurchschnitt'
        }
    translated_list = [columns_german.get(item, item) for item in feature_list]

    return translated_list


class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
        return np.argmax(output.cpu().numpy(), axis=1)


def measure_importance(model, X_test, y_test):
    model_sklearn = SklearnWrapper(model, device)
    model.eval()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    r = permutation_importance(model_sklearn, X_test_np, y_test_np, n_repeats=30)
    return r


def import_data():
    df = pd.read_csv(os.path.join('input', 'corporateCreditRatingWithFinancialRatios.csv'))
    df['index']=df.index
    # df = df.loc[df['Rating Agency'] == "Standard & Poor's Ratings Services"]

    rating_mapping = {
        'AAA': 'Sehr Hohe Bonität', 'AA+': 'Sehr Hohe Bonität', 'AA': 'Sehr Hohe Bonität', 'AA-': 'Sehr Hohe Bonität', 
        'A+': 'Gute Bonität', 'A': 'Gute Bonität', 'A-': 'Gute Bonität', 
        'BBB+': 'Befriedigende Bonität', 'BBB': 'Befriedigende Bonität', 'BBB-': 'Befriedigende Bonität', 
        'BB+': 'Angespannte Bonität', 'BB': 'Angespannte Bonität', 'BB-': 'Angespannte Bonität',
        'B+': 'Mangelhafte Bonität', 'B': 'Mangelhafte Bonität', 'B-': 'Mangelhafte Bonität',
        'CCC+': 'Ungenügende Bonität', 'CCC': 'Ungenügende Bonität', 'CCC-': 'Ungenügende Bonität', 
        'CC': 'Ungenügende Bonität', 'C': 'Ungenügende Bonität', 
        'D': 'Insolvent'
    }

    df['Rating Category'] = df['Rating'].map(rating_mapping)
    df = df[df['Rating Category'].notna()]
    df = df.loc[df['Rating Category']!="Insolvent"]

    encoder = LabelEncoder()
    df['Sector'] = encoder.fit_transform(df.Sector.values)

    df['Rating Category Encoded'] = df['Rating Category'].apply(lambda x: ['Sehr Hohe Bonität', 'Gute Bonität', 
                                                                        'Befriedigende Bonität', 'Angespannte Bonität', 
                                                                        'Mangelhafte Bonität', 'Ungenügende Bonität'].index(x))
    
    return df

class RatingsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(11, 256)
        self.hidden2 = nn.Linear(256, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 6)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.08, inplace=False)

    def forward(self, x):
        x = self.tanh(self.hidden1(x))
        x = self.dropout(x)
        x = self.tanh(self.hidden2(x))
        x = self.dropout(x)
        x = self.tanh(self.hidden3(x))
        x = self.dropout(x)
        x = self.tanh(self.hidden4(x))
        x = self.dropout(x)
        x = self.softmax(self.output(x))
        return x


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_accuracy = -np.inf

    def early_stop(self, accuracy):
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.counter = 0
        elif accuracy < (self.max_accuracy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    # assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3070 Ti Laptop GPU'

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Model will run on {device}.")

    torch.manual_seed(42)


    cols_to_keep = ['Current Ratio',
    'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
    'EBIT Margin', 'Asset Turnover',
    'ROE - Return On Equity', 'Return On Tangible Equity',
    'Operating Cash Flow Per Share', 'Free Cash Flow Per Share', 'Sector']
    
    df = import_data()


    X = df[cols_to_keep]
    y = df.loc[:, df.columns == 'Rating Category Encoded'].to_numpy()

    onehot = OneHotEncoder(sparse_output=False).fit(y)
    y = onehot.transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)

    model = RatingsNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, min_lr=0.001)
    stopper = EarlyStopper(patience=100, min_delta=0)

    n_epochs = 300
    batch_size = 8
    batches_per_epoch = len(X_train)//batch_size

    best_acc = -np.inf
    best_weights = None

    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        epoch_mse = []

        model.train()
        with tqdm.trange(batches_per_epoch, unit='batch', mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch+1}/{n_epochs}")
            for i in bar:
                start = i * batch_size
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                y_pred = model(X_batch).to(device)
                loss = loss_fn(y_pred, y_batch)

                mse_loss = mse_loss_fn(y_pred, y_batch)
                epoch_mse.append(float(mse_loss))

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc), mse=float(mse_loss))


        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)

        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_lr = optimizer.param_groups[0]['lr']
            torch.save(model.state_dict(), os.path.join('torch_weights', 'best_weights.pt'))
        print(f"Epoch {epoch+1} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%, Lr={optimizer.param_groups[0]['lr']}")
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(acc)
        after_lr = optimizer.param_groups[0]['lr']
        if stopper.early_stop(acc):
            print('Model stopped by early stopping.')
            break

    model.load_state_dict(torch.load(os.path.join('torch_weights', 'best_weights.pt')))
    print(f'Die höchste Genauigkeit ist: {round(best_acc*100, 5)}% with learning rate {best_lr}')

    plt.figure()
    plt.plot(train_loss_hist, label='Training')
    plt.plot(test_loss_hist, label='Test')
    plt.xlabel('Epochen')
    plt.ylabel('Kreuzentropie-Verlust')

    plt.savefig(os.path.join('torch_eval', 'test_train_loss.png'))

    plt.figure()
    plt.plot(train_acc_hist, label='Training')
    plt.plot(test_acc_hist, label='Test')
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit')
    plt.legend()
    plt.savefig(os.path.join('torch_eval', 'test_train_acc.png'))

    y_pred = model(X_test)
    y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
    y_test = np.argmax(y_test.cpu(), axis=1)
    

    sorted = ['Sehr Hohe Bonität',
    'Gute Bonität',
    'Befriedigende Bonität',
    'Angespannte Bonität',
    'Mangelhafte Bonität',
    'Ungenügende Bonität']

    translate = {
    0: 'Sehr Hohe Bonität',
    1: 'Gute Bonität',
    2: 'Befriedigende Bonität',
    3: 'Angespannte Bonität',
    4: 'Mangelhafte Bonität',
    5: 'Ungenügende Bonität'
    }

    y_test_uncoded = np.vectorize(translate.get)(y_test)
    y_pred_uncoded = np.vectorize(translate.get)(y_pred)

    cf_matrix = confusion_matrix(y_test, y_pred)
    df_matrix = pd.DataFrame(cf_matrix).rename(index=translate, columns=translate)
    df_matrix.to_csv(os.path.join('torch_eval', 'cf_matrix.csv'), sep=';', decimal=',')


    disp = ConfusionMatrixDisplay.from_predictions(y_test_uncoded, y_pred_uncoded, labels=sorted).plot(xticks_rotation=90)
    
    plt.savefig(os.path.join('torch_eval', 'cf_matrix_display.png'), bbox_inches='tight')

    report = classification_report(y_test_uncoded, y_pred_uncoded, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    df_columns = {'precision': 'Präzision', 'recall': 'Sensitivität', 'f1-score': 'F1-Score', 'support': 'Testdatensätze'}
    df_index = {'accuracy': 'Genauigkeit', 'macro avg': 'Ungewichteter Durchschnitt', 'weighted avg': 'Gewichteter Durchschnitt'}

    df_report = df_report.rename(columns=df_columns, index=df_index)

    df_report['Testdatensätze'] = df_report['Testdatensätze'].astype(int)

    for value in df_index.values():
        sorted.append(value)


    df_report = df_report.reindex(sorted)
    features_german = translate_features(cols_to_keep)

    df_report.to_excel(os.path.join('torch_eval', 'classification_report.xlsx'))

    print(f'The shape of X_test is: {X_test.shape}')
    print(f'The shape of y_test is: {y_test.shape}')

    r = measure_importance(model, X_test, y_test)

    importances = r.importances_mean
    std = r.importances_std
    indices = np.argsort(importances)[::-1]

    importances = np.round(importances, decimals=3)

    plt.figure()
    plt.bar(range(X_test.shape[1]), importances[indices], color="grey", align="center")
    plt.xticks(range(X_test.shape[1]), features_german, rotation='vertical')
    plt.xlim([-1, X_test.shape[1]])
    ax = plt.subplot()
    plt.bar_label(ax.containers[0])
    plt.savefig(os.path.join('torch_eval', 'permutation_importance.png'), bbox_inches='tight')

