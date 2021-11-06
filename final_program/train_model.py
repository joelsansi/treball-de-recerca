# package importing
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


# arguments for required files
args = {'embeddings': 'output/embeddings.pickle', 
'recognizer': 'output/recognizer.pickle', 
'le': 'output/le.pickle'}

# load embeddings
print("[INFO] carregant embeddings ...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] codificant etiquetes ...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train model using the 128-d vectors
print("[INFO] entrenant model 😎 ...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# save recognition model
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# save label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()