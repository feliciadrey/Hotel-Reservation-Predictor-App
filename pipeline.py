#OOP Code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pickle
import gzip

class HotelClassifier:
    def __init__(self, filepath, seed=42):
        self.filepath = filepath
        self.seed = seed
    #Method untuk load dan cleaning data
    def load_data(self):
        df = pd.read_csv(self.filepath)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop('Booking_ID', axis=1, inplace=True)
        self.df = df
    #Visualisasi distribusi fitur numerik
    def check_dist(self, df, column):
        for i in column:
            plt.figure(figsize=(10,2))
            plt.subplot(1,2,1)
            sns.histplot(df[i], bins=20)
            plt.title(f'Histogram of {i}')
            plt.subplot(1,2,2)
            sns.boxplot(y=df[i])
            plt.title(f'Boxplot of {i}')
            plt.show()
    #Visualisasi distribusi variabel target
    def visualize_target(self):
        plt.figure(figsize=(9, 5))
        sns.countplot(x='booking_status', data=self.df, order=self.df['booking_status'].value_counts().index)
        plt.xlabel('Booking Status')
        plt.ylabel('Count')
        plt.title('Distribution of Booking Status (Target)')
        plt.show()
    #Preprocessing: label encoding, onehot encoding, split data, resampling
    def preprocess(self):
        cat, num = [], []
        for i in self.df.columns:
            if self.df[i].dtype == 'object':
                cat.append(i)
            else:
                num.append(i)
        self.categorical = cat
        self.numerical = num

        le = LabelEncoder()
        self.df['booking_status'] = le.fit_transform(self.df['booking_status'])

        self.visualize_target()
        self.check_dist(self.df[num], num)

        x = self.df.drop(columns=['booking_status'])
        y = self.df['booking_status']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed)

        scaler = RobustScaler()
        x_train[num] = scaler.fit_transform(x_train[num])
        x_test[num] = scaler.transform(x_test[num])

        cat.remove('booking_status') if 'booking_status' in cat else None

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc_train = ohe.fit_transform(x_train[cat])
        x_train_enc = pd.DataFrame(enc_train, columns=ohe.get_feature_names_out(cat))
        x_train = x_train.reset_index(drop=True).drop(cat, axis=1)
        x_train = pd.concat([x_train, x_train_enc], axis=1)

        enc_test = ohe.transform(x_test[cat])
        x_test_enc = pd.DataFrame(enc_test, columns=ohe.get_feature_names_out(cat))
        x_test = x_test.reset_index(drop=True).drop(cat, axis=1)
        x_test = pd.concat([x_test, x_test_enc], axis=1)

        ros = RandomOverSampler(random_state=self.seed)
        x_train, y_train = ros.fit_resample(x_train, y_train)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.ohe = ohe
        self.le = le
    #Training best model (Random Forest)
    def train_model(self):
        model = RandomForestClassifier(random_state=self.seed, max_depth=30)
        model.fit(self.x_train, self.y_train)
        self.model = model
    #Evaluasi model
    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")

        #AUC
        y_pred_proba = self.model.predict_proba(self.x_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve Using Random Forest')
        plt.legend(loc='lower right')
        plt.show()
    #Save model
    def save_model(self, filename='rfmodel.pkl'):
        with gzip.open('rfmodel_compressed.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('oh_encoder.pkl', 'wb') as f:
            pickle.dump(self.ohe, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.le, f)

    def run(self):
        self.load_data()
        self.preprocess()
        self.train_model()
        self.evaluate()
        self.save_model()

#Run model yang telah dibuat
model = HotelClassifier("Dataset_B_hotel.csv")
model.run()
