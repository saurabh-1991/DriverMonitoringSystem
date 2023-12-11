import matplotlib.pyplot as plt
import skimage
from PIL import Image
import random as rnd
import seaborn as sns
from data_prepare import classes
def visualize_train_data(train_data):
    print('[Visualize Data]:-\n')
    plt.figure(figsize=(60, 8))
    for idx, i in enumerate(train_data.classname.unique()):
        plt.subplot(4, 7, idx + 1)
        df = train_data[train_data['classname'] == i].reset_index(drop=True)
        image_path = df.loc[rnd.randint(0, len(df)) - 1, 'path']
        img = Image.open(image_path)
        img = img.resize((224, 224))
        plt.imshow(img)
        plt.axis('off')
        plt.title(classes[i])
    plt.tight_layout()
    plt.show()

def visualize_class_distribution_analysis(train_data):
  plot = sns.countplot(x=train_data['classname'], color='#2596be')
  sns.set(rc={'figure.figsize': (30, 25)})
  sns.despine()
  plot.set_title('Class Distribution\n', font='serif', x=0.1, y=1, fontsize=18);
  plot.set_ylabel("Count", x=0.02, font='serif', fontsize=12)
  plot.set_xlabel("Driver classes", fontsize=15, font='serif')
  for p in plot.patches:
      plot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center', va='center', xytext=(0, -20), font='serif', textcoords='offset points', size=15)
  plt.figure(figsize=(5, 5))
  class_cnt = train_data.groupby(['classname']).size().reset_index(name='counts')
  colors = sns.color_palette('Paired')[0:9]
  plt.pie(class_cnt['counts'], labels=class_cnt['classname'], colors=colors, autopct='%1.1f%%')
  plt.legend(loc='upper right')
  plt.show()