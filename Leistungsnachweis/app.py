import gradio as gr
import pandas as pd
import pickle

# Modell laden
model_filename = "feature_engineering.pkl"  # Falls du das Modell gespeichert hast
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

def predict_price(rooms, area, postalcode, pop, pop_dens, tax_income, density_factor):
    input_data = pd.DataFrame([[rooms, area, postalcode, pop, pop_dens, tax_income, density_factor]],
                              columns=['rooms', 'area', 'postalcode','pop', 'pop_dens', 'tax_income','density_factor'])
    price = model.predict(input_data)[0]
    return f"Geschätzter Preis: {price:.2f} CHF"

app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Anzahl Zimmer"),
        gr.Number(label="Fläche (qm)"),
        gr.Number(label="Postleitzahl"),
    ],
    outputs="text",
    title="Apartment Preis Schätzer",
    description="Geben Sie die Anzahl der Zimmer, die Fläche und die Postleitzahl ein. Bevölkerungsdaten werden automatisch abgerufen und density_factor berechnet.",
    examples = [[2, 122, 8050],
            [1.5, 30, 8008]]
)


# Launch the Gradio app
app.launch(share=True)
