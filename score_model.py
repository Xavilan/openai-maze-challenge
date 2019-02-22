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
    team_name = Column(String(250))
    score = Column(Float, default=0)
    overall_score = Column(Float, default=0)
    extra = Column(String(250))
    param_int1 = Column(Integer)
    param_int2 = Column(Integer)
    param_int3 = Column(Integer)
    param_str1 = Column(String(250))
    param_str2 = Column(String(250))
    param_str3 = Column(String(250))

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
            'team_name': self.team_name,
            'score': round(self.score, 4),
            'overall_score': round(self.overall_score, 4),
            'extra': self.extra,
            'param_int1': str(self.param_int1),
            'param_int2': str(self.param_int2),
            'param_int3': str(self.param_int3),
            'param_str1': str(self.param_str1),
            'param_str2': str(self.param_str2),
            'param_str3': str(self.param_str3),
            'created_at': str(self.created_at)
        }


ins = inspect(engine)
if len(ins.get_table_names()) <= 0:
    print('Create database')
    Base.metadata.create_all(bind=engine)

Session = sessionmaker()
Session.configure(bind=engine)
session = Session()
