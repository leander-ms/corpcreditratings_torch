import tkinter as tk
import pandas as pd
import torch
import joblib
import os
from mlp_torch2_esg import RatingsNet
import numpy as np


def load_model_and_scaler():
    model = RatingsNet(input_shape=13)
    model.load_state_dict(torch.load(os.path.join('torch_weights', 'best_weights_esg.pt')))
    scaler = joblib.load(os.path.join('torch_weights', 'model2_scaler.pkl'))
    encoder = joblib.load(os.path.join('torch_weights', 'model2_encoder.pkl'))

    return model, scaler, encoder


def get_sector_mean(sector):
    return {
        'BusEq': 17.696725,
        'Chems': 29.170629,
        'Durbl': 17.045405,
        'Enrgy': 36.465781,
        'Hlth': 25.559804,
        'Manuf': 25.751043,
        'Money': 18.810945,
        'NoDur': 23.492599,
        'Other': 24.341715,
        'Shops': 19.554717,
        'Telcm': 23.198681,
        'Utils': 28.493429
    }[sector]


def predict_rating(model, scaler, encoder, features):
    input_data = pd.DataFrame()
    
    for header, value in features.items():
        input_data[header] = [value]
    
    esg_to_sector = features.get('ESG Rating')/get_sector_mean(features.get('Sector'))
    input_data.insert(11, 'ESG to Sector Average', [esg_to_sector])

    input_data_cont = scaler.transform(input_data.iloc[:, 0:12])
    input_data_cat = encoder.transform(input_data.iloc[:, 12:13])
    input_data_complete = np.append(input_data_cont, input_data_cat.reshape(-1, 1), axis=1)

    model.eval()
    with torch.no_grad():
        predicted_rating_index = model(torch.tensor(input_data_complete, dtype=torch.float32))
    predicted_rating_index = np.argmax(predicted_rating_index.detach().numpy(), axis=1)

    translate_rating_index = {0: 'Sehr Hohe Bonität', 1: 'Gute Bonität', 2: 'Befriedigende Bonität', 3: 'Angespannte Bonität', 4: 'Mangelhafte Bonität', 5: 'Ungenügende Bonität'}

    predicted_rating = translate_rating_index.get(predicted_rating_index[0])

    return predicted_rating


def features_to_dict(**kwargs):
    feature_dict = {}
    for key, value in kwargs.items():
        appended_dict = {key:value}
        feature_dict.update(appended_dict)

    return feature_dict


if __name__ == '__main__':
    window = tk.Tk()
    window.title('Predict Rating')

    inputs_frame = tk.Frame(window)
    inputs_frame.pack(pady=20)

    current_ratio_label = tk.Label(inputs_frame, text='Current Ratio:')
    current_ratio_label.grid(row=0, column=0, padx=5, pady=5)

    current_ratio_var = tk.StringVar()
    current_ratio_entry = tk.Entry(inputs_frame, textvariable=current_ratio_var)
    current_ratio_entry.grid(row=0, column=1, padx=5, pady=5)


    debt_capital_label = tk.Label(inputs_frame, text='Debt Capital:')
    debt_capital_label.grid(row=1, column=0, padx=5, pady=5)
    debt_capital_var = tk.StringVar()
    debt_capital_entry = tk.Entry(inputs_frame, textvariable=debt_capital_var)
    debt_capital_entry.grid(row=1, column=1, padx=5, pady=5)

    debt_equity_label = tk.Label(inputs_frame, text='Debt Equity:')
    debt_equity_label.grid(row=2, column=0, padx=5, pady=5)
    debt_equity_var = tk.StringVar()
    debt_equity_entry = tk.Entry(inputs_frame, textvariable=debt_equity_var)
    debt_equity_entry.grid(row=2, column=1, padx=5, pady=5)

    gross_margin_label = tk.Label(inputs_frame, text='Gross Margin:')
    gross_margin_label.grid(row=3, column=0, padx=5, pady=5)
    gross_margin_var = tk.StringVar()
    gross_margin_entry = tk.Entry(inputs_frame, textvariable=gross_margin_var)
    gross_margin_entry.grid(row=3, column=1, padx=5, pady=5)

    ebit_margin_label = tk.Label(inputs_frame, text='EBIT Margin:')
    ebit_margin_label.grid(row=4, column=0, padx=5, pady=5)
    ebit_margin_var = tk.StringVar()
    ebit_margin_entry = tk.Entry(inputs_frame, textvariable=ebit_margin_var)
    ebit_margin_entry.grid(row=4, column=1, padx=5, pady=5)

    asset_turnover_label = tk.Label(inputs_frame, text='Asset Turnover:')
    asset_turnover_label.grid(row=5, column=0, padx=5, pady=5)
    asset_turnover_var = tk.StringVar()
    asset_turnover_entry = tk.Entry(inputs_frame, textvariable=asset_turnover_var)
    asset_turnover_entry.grid(row=5, column=1, padx=5, pady=5)

    return_equity_label = tk.Label(inputs_frame, text='Return on Equity:')
    return_equity_label.grid(row=6, column=0, padx=5, pady=5)
    return_equity_var = tk.StringVar()
    return_equity_entry = tk.Entry(inputs_frame, textvariable=return_equity_var)
    return_equity_entry.grid(row=6, column=1, padx=5, pady=5)

    return_tangible_equity_label = tk.Label(inputs_frame, text='Return on Tangible Equity:')
    return_tangible_equity_label.grid(row=7, column=0, padx=5, pady=5)
    return_tangible_equity_var = tk.StringVar()
    return_tangible_equity_entry = tk.Entry(inputs_frame, textvariable=return_tangible_equity_var)
    return_tangible_equity_entry.grid(row=7, column=1, padx=5, pady=5)

    operating_cf_pershare_label = tk.Label(inputs_frame, text='Operating CF Per Share:')
    operating_cf_pershare_label.grid(row=8, column=0, padx=5, pady=5)
    operating_cf_pershare_var = tk.StringVar()
    operating_cf_pershare_entry = tk.Entry(inputs_frame, textvariable=operating_cf_pershare_var)
    operating_cf_pershare_entry.grid(row=8, column=1, padx=5, pady=5)

    fcf_pershare_label = tk.Label(inputs_frame, text='Free Cash Flow Per Share:')
    fcf_pershare_label.grid(row=9, column=0, padx=5, pady=5)
    fcf_pershare_var = tk.StringVar()
    fcf_pershare_entry = tk.Entry(inputs_frame, textvariable=fcf_pershare_var)
    fcf_pershare_entry.grid(row=9, column=1, padx=5, pady=5)

    esg_rating_label = tk.Label(inputs_frame, text='ESG Rating:')
    esg_rating_label.grid(row=10, column=0, padx=5, pady=5)
    esg_rating_var = tk.StringVar()
    esg_rating_entry = tk.Entry(inputs_frame, textvariable=esg_rating_var)
    esg_rating_entry.grid(row=10, column=1, padx=5, pady=5)

    sector_label = tk.Label(inputs_frame, text='Sector:')
    sector_label.grid(row=12, column=0, padx=5, pady=5)
    sector_var = tk.StringVar()
    sector_entry = tk.Entry(inputs_frame, textvariable=sector_var)
    sector_entry.grid(row=12, column=1, padx=5, pady=5)

    # Button to execute prediction
    def on_predict():
        model, scaler, encoder = load_model_and_scaler()
        features = {
            'Current Ratio': float(current_ratio_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Long-term Debt / Capital': float(debt_capital_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Debt/Equity Ratio': float(debt_equity_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Gross Margin': float(gross_margin_var.get().replace(',', '.').rstrip('\n').strip('')),
            'EBIT Margin': float(ebit_margin_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Asset Turnover': float(asset_turnover_var.get().replace(',', '.').rstrip('\n').strip('')),
            'ROE - Return On Equity': float(return_equity_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Return On Tangible Equity': float(return_tangible_equity_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Operating Cash Flow Per Share': float(operating_cf_pershare_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Free Cash Flow Per Share': float(fcf_pershare_var.get().replace(',', '.').rstrip('\n').strip('')),
            'ESG Rating': float(esg_rating_var.get().replace(',', '.').rstrip('\n').strip('')),
            'Sector': sector_var.get()
        }

        predicted_rating = predict_rating(model, scaler, encoder, features)
        predicted_rating_var.set(predicted_rating)

    predict_button = tk.Button(window, text="Predict Rating", command=on_predict)
    predict_button.pack(pady=20)

    # Label to display the prediction
    predicted_rating_var = tk.StringVar()
    predicted_rating_label = tk.Label(window, textvariable=predicted_rating_var)
    predicted_rating_label.pack(pady=20)

    window.mainloop()
