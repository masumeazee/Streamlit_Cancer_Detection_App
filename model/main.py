import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_data():
    data = pd.read_csv('data/cancer_data.csv')
    print(data.head())
    return data


def create_model(data):
    X = data.drop(['target'], axis=1)
    y = data['target']

    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    # split data
    X_trian, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # train model
    model = LogisticRegression()
    model.fit(X_trian, y_train)

    # test model
    y_pred = model.predict(X_test)
    print("accuracy of model :", accuracy_score(y_test, y_pred))
    print("classification report :\n", classification_report(y_test, y_pred))
    return model, scaler


def main():
    data = get_data()
    model, scaler = create_model(data)

    # write binary file with pickle5
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
