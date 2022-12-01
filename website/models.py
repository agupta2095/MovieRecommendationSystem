from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func
import pandas as pd
from website.recommedation.utils import read_pickle, grab_highest_rated


class Watched(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    data = db.Column(db.String(1000))





class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


    data=db.Column(db.String(1000000))



class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    rated=db.relationship("Watched")
    recommended=db.relationship("Recommendation")
