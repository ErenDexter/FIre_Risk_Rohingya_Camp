import matplotlib.pyplot as plt
import pandas as pd

training_dataset = pd.read_csv('19_10_23_Master_Dataset.csv')

#font_properties = {'fontname': 'Arial', 'fontsize': 14}

plt.hist(training_dataset['CSI'],
         alpha = 0.7,
         bins = 45, 
         edgecolor='white',
         label='Pre Fire CSI') 
plt.axvline(training_dataset['CSI'].mean(), color='k', linestyle='dashed', linewidth=1)
  
plt.hist(training_dataset['post_CSI'], 
         alpha = 0.7, 
         bins = 35,
         edgecolor='white', 
         label='Post Fire CSI') 
plt.axvline(training_dataset['post_CSI'].mean(), color='k', linestyle='dashed', linewidth=1)
#plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

plt.xticks(fontsize=12, fontname='Arial', weight='bold')
plt.yticks(fontsize=12, fontname='Arial', weight='bold')  

plt.legend(loc='upper right') 
plt.title('CSI Pre-Post Frequency') 
plt.show()