import base64
import torch
from torch import nn
from io import BytesIO
from matplotlib.figure import Figure
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

class Model(nn.Module):
    def __init__(self, noise_height, noise_width):
        super().__init__()
        self.noise_height = noise_height
        self.noise_width = noise_width
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(noise_height * noise_width, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



model = Model(
    noise_height=1024,
    noise_width=1024,
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.db'
app.config['SQLALCHEMY_TRACK_MODIFICTIONS'] = False
db = SQLAlchemy(app)

class DataBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.Integer, nullable=True)
    noise_height = db.Column(db.Integer, nullable=True)
    noise_width = db.Column(db.Integer, nullable=True)

@app.route("/")
def page():
    X = torch.rand(1, model.noise_height, model.noise_width)
    
    logits = model(X)
    pred_prohab = nn.Softmax(dim=1)(logits)
    y_pred = pred_prohab.argmax(1)
    
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(X.cpu().squeeze(), cmap="gray")
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    new_db_entry = DataBase(
        prediction = int(y_pred.item()),
        noise_height = model.noise_height,
        noise_width = model.noise_width,
    )
    
    db.session.add(new_db_entry)
    db.session.commit()
    
    return render_template('index.html', prediction=y_pred, plot=plot)

if __name__ == "__main__":
    app.run(debug=True)