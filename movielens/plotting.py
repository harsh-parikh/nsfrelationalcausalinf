import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

fpr, tpr, threshold = roc_curve(test_target, var_output.detach().numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'blue', label = 'Embd, AUC = %0.2f' % roc_auc)


plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
