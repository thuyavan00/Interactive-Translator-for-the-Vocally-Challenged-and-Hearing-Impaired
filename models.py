from db import db



class forumdb(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String, nullable=False)
    title= db.Column(db.String, nullable=False)
    text = db.Column(db.String, nullable=False)
    img = db.Column(db.LargeBinary(length=2048))
    mimetype = db.Column(db.Text, nullable=False)