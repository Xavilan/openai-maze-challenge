# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, inspect, Column, String
from sqlalchemy import Integer, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

db_name = 'db.sqlite'
db_url = "sqlite:///web/" + db_name
engine = create_engine(db_url)

Base = declarative_base()


class Score(Base):
    __tablename__ = 'Score'
    id = Column(Integer, primary_key=True, autoincrement=True)
    runtime = Column(String(250))
    round_no = Column(Integer)
    title = Column(String(250))
    n_bandits = Column(Integer)
    b_alpha = Column(Integer)
    b_beta = Column(Integer)
    score = Column(Float, default=0)
    overal_score = Column(Float, default=0)
    extra = Column(String(250))

    created_at = Column(DateTime(), default=datetime.datetime.utcnow)

    def __init__(self, *args, **kwargs):
        super(Score, self).__init__(*args, **kwargs)
        self.created_at = datetime.datetime.utcnow()

    @property
    def json(self):
        return {
            'id': str(self.id),
            'round_no': str(self.round_no),
            'runtime': self.runtime,
            'title': self.title,
            'n_bandits': self.n_bandits,
            'b_alpha': round(self.b_alpha, 6),
            'b_beta': round(self.b_beta, 6),
            'score': round(self.score, 4),
            'overal_score': round(self.overal_score, 4),
            'extra': self.extra,
            'created_at': str(self.created_at)
        }


ins = inspect(engine)
if len(ins.get_table_names()) <= 0:
    print('Create database')
    Base.metadata.create_all(bind=engine)

Session = sessionmaker()
Session.configure(bind=engine)
session = Session()
