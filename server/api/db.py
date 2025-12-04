from electrus.asynchronous import Electrus

client = Electrus()
database = client["ResumeAnalyser"]

collection = database["ResumeMetaCollection"]