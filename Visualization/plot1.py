import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('English11_PostProcessing.csv')

plt.hist(data['Vocab Label'],normed=1, color='green')
plt.title("Vocabulary Section")
plt.xlabel("Label")
plt.ylabel("No. of Students")
plt.show()

plt.hist(data['Grammar Label'])
plt.title("Grammar Section")
plt.xlabel("Label")
plt.ylabel("No. of Students")
plt.show()

plt.hist(data['Reading Label'])
plt.title("Reading Section")
plt.xlabel("Label")
plt.ylabel("No. of Students")
plt.show()

plt.hist(data['Computer Label'])
plt.title("Computer Section")
plt.xlabel("Label")
plt.ylabel("No. of Students")
plt.show()

plt.hist(data['Writing Label'])
plt.title("Writing Section")
plt.xlabel("Label")
plt.ylabel("No. of Students")
plt.show()