import pandas as pd
from sklearn.preprocessing import LabelEncoder , OrdinalEncoder

# dikartenakan data yang saya tidak punya kategorial jadi bikin data sendiri
data_contoh = {
    'Kota': ['jakarta', 'Bandung', 'Surabaya', 'Bandung', 'Jakarta'],
    'Pendidikan': ['SMA', 'S1', 'SMP', 'SMA', 'S1'],
    'Gaji': [10, 15, 7, 11, 16]
}
df = pd.DataFrame(data_contoh)
print("Contoh awal")
print(df)
print("\n" + "="*40 + "\n")

# mengonversi tipe data
category_feacture = df.select_dtypes(include=['object']).columns
print("fitur kategorial")
print(df[category_feacture])
print("\n" + "="*40 + "\n")

# one hot encoding ===================================================================
print("one hot encoding 'kota'")
df_one_hot = pd.get_dummies(df, columns=category_feacture)

# data setelah one hot encoding
print("setelah one hot encoding")
print(df_one_hot)
print("\n" + "="*40 + "\n")

# label encoding ===================================================================
print("label encoding 'kota'")
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_lencoder = df.copy()

df_lencoder['Kota'] = label_encoder.fit_transform(df_lencoder['Kota'])

print("after label encoding")
print(df_lencoder)
print("\n" + "="*40 + "\n")

# Ordinal encoding ===================================================================
print("ordinal encoding 'pendidikan'")
tingkat_pendidikan = ['SMP', 'SMA', 'S1']

ordinal_encoder = OrdinalEncoder(categories=[tingkat_pendidikan])
df_oencoder = df.copy()

df_oencoder['Pendidikan'] = ordinal_encoder.fit_transform(df_oencoder[['Pendidikan']])

print("after ordinal encoding")
print(df_oencoder)
print("\n" + "="*40 + "\n")
