import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def getAvg(run1, run2, run3):
  array_avg = []
  for i in range(0, len(run1)):
    avg = (run1[i] + run2[i] + run3[i]) / 3.0
    array_avg.append(avg)
  return array_avg

# firstOrder_T1_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT1/firstOrderT1_run1/firstOrderT1_run1_testAcc.npy")
# firstOrder_T1_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT1/firstOrderT1_run2/firstOrderT1_run2_testAcc.npy")
# firstOrder_T1_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT1/firstOrderT1_run3/firstOrderT1_run3_testAcc.npy")
#
# secondOrder_T1_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT1/secondOrderT1_run1_testAcc.npy")
# secondOrder_T1_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT1/secondOrderT1_run2_testAcc.npy")
# secondOrder_T1_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT1/secondOrderT1_run3_testAcc.npy")
#
# firstOrder_T5_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT5/firstOrderT5_run1/firstOrderT5_run1_testAcc.npy")
# firstOrder_T5_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT5/firstOrderT5_run2/firstOrderT5_run2_testAcc.npy")
# firstOrder_T5_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT5/firstOrderT5_run3/firstOrderT5_run3_testAcc.npy")
#
# secondOrder_T5_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT5/secondOrderT5_run1/secondOrderT5_run1_testAcc.npy")
# secondOrder_T5_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT5/secondOrderT5_run3/secondOrderT5_run5_testAcc.npy")
# secondOrder_T5_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT5/secondOrderT5run2/secondOrderT5_run2_testAcc.npy")
#
# firstOrder_T10_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT10/firstOderT10_run1/firstOrderT10_run1_testAcc.npy")
# firstOrder_T10_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT10/firstOrderT10_run2/firstOrderT10_run2_testAcc.npy")
# firstOrder_T10_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderT10/firstOrderT10_run3/firstOrderT10_run3_testAcc.npy")
#
# secondOrder_T10_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT10/secondOrderT10_run1/secondOrderT10_run1_testAcc.npy")
# secondOrder_T10_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT10/secondOrderT10_run2/secondOrderT10_run2_testAcc.npy")
# secondOrder_T10_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT10/secondOrderT10_run3/secondOrderT10_run3_testAcc.npy")
#
# # get the average of the three runs
# firstOrder_T1 = getAvg(firstOrder_T1_run1, firstOrder_T1_run2, firstOrder_T1_run3)
# firstOrder_T5 = getAvg(firstOrder_T5_run1, firstOrder_T5_run2, firstOrder_T5_run3)
# firstOrder_T10 = getAvg(firstOrder_T10_run1, firstOrder_T10_run2, firstOrder_T10_run3)
# secondOrder_T1 = getAvg(secondOrder_T1_run1, secondOrder_T1_run2, secondOrder_T1_run3)
# secondOrder_T5 = getAvg(secondOrder_T5_run1, secondOrder_T5_run2, secondOrder_T5_run3)
# secondOrder_T10 = getAvg(secondOrder_T10_run1, secondOrder_T10_run2, secondOrder_T10_run3)
#
# # Plot for empirical question 2
# t1 = np.repeat('T=1', 1000)
# t5 = np.repeat('T=5', 1000)
# t10 = np.repeat('T=10', 1000)
# first = np.repeat('First Order MAML', 3000)
# second = np.repeat('Second Order MAML', 3000)
#
# # Plot
# data = pd.DataFrame({
#     'Accuracy': np.concatenate([firstOrder_T1, firstOrder_T5, firstOrder_T10,
#                                 secondOrder_T1, secondOrder_T5, secondOrder_T10]),
#     'T': np.concatenate([t1, t5, t10, t1, t5, t10]),
#     'MAML Order': np.concatenate([first, second])
# })

test_new = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soTrue-domniglot-T1-is28-rFalse-dNone-s0/test-accuracy.npy")
test_original = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderT1/secondOrderT1_run1_testAcc.npy")
t1 = np.repeat('T=1', 2000)
second = np.repeat('Second Order MAML', 2000)
data = pd.DataFrame({
    'Accuracy': np.concatenate([test_new, test_original]),
    'T': t1,
    'MAML Order': second
})

# Plotting box plots
sns.boxplot(data=data, x="T", y="Accuracy", hue="MAML Order")

# Adding labels and title
plt.xlabel('T')
plt.ylabel('Test Accuracy')
plt.title('Box Plot of Test Accuracy for Different T values and MAML Orders')

plt.grid()
# Show the plot
plt.show()

# Plot for empirical question 4

# firstOrderB4_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderB4/firstOrderB4_run1/firstOrderB4_run1_testAcc.npy")
# firstOrderB4_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderB4/firstOrderB4_run2/firstOrderB4_run2_testAcc.npy")
# firstOrderB4_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/firstOrderB4/FirstOrderB4_run3/test-accuracy.npy")
#
# secondOrderB4_run1 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderB4/secondOrderB4run1/test-accuracy.npy")
# secondOrderB4_run2 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderB4/secondOrderB4_run2/test-accuracy.npy")
# secondOrderB4_run3 = np.load("/Users/giuliarivetti/Desktop/Leiden Univ/Automated Machine Learning/Assignments/Assignment 2/a2/results/secondOrderB4/secondOrderB4_run3/test-accuracy.npy")
#
# firstOrderB4 = getAvg(firstOrderB4_run1, firstOrderB4_run2, firstOrderB4_run3)
# secondOrderB4 = getAvg(secondOrderB4_run1, secondOrderB4_run2, secondOrderB4_run3)
# firstOrderB1 = firstOrder_T1
# secondOrderB1 = secondOrder_T1
#
# window_size = 40
#
# plt.figure(figsize=(10, 5))
#
# firstOrderB1 = np.convolve(firstOrderB1, np.ones(window_size) / window_size, mode='valid')
# firstOrderB4 = np.convolve(firstOrderB4, np.ones(window_size) / window_size, mode='valid')
# secondOrderB1 = np.convolve(firstOrderB1, np.ones(window_size) / window_size, mode='valid')
# secondOrderB4 = np.convolve(secondOrderB4, np.ones(window_size) / window_size, mode='valid')
#
# # Plotting the smoothed test accuracy lines for First-order and Second-order
# plt.plot(firstOrderB1, label='FO B1')
# plt.plot(firstOrderB4, label='FO B4')
# plt.plot(secondOrderB1, label='SO B1', linestyle='--')
# plt.plot(secondOrderB4, label='SO B4', linestyle='--')
#
# plt.xlabel('Number of tasks evaluation')
# plt.ylabel('Accuracy')
# plt.title('Meta-test Accuracy over tasks evaluations')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()
#
#
#
# # Plot for empirical question 1
# # Load the training and validation loss data for three runs for SO
# train_losses_so_1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN1/train-loss1.npy")
# val_losses_so_1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN1/val-loss1.npy")
#
# train_losses_so_2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN2/train-loss2.npy")
# val_losses_so_2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN2/val-loss2.npy")
#
# train_losses_so_3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN3/train-loss3.npy")
# val_losses_so_3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN3/val-loss3.npy")
#
# # Load the training and validation loss data for three runs for FO
# train_losses_fo_1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run1/firstOrderT1_run1_trainLoss.npy")
# val_losses_fo_1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run1/firstOrderT1_run1_valLoss.npy")
#
# train_losses_fo_2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run2/firstOrderT1_run2_trainLoss.npy")
# val_losses_fo_2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run2/firstOrderT1_run2_valLoss.npy")
#
# train_losses_fo_3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run3/firstOrderT1_run3_trainLoss.npy")
# val_losses_fo_3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run3/firstOrderT1_run3_valLoss.npy")
#
# # Calculate average losses for SO
# avg_train_losses_so = np.mean([train_losses_so_1, train_losses_so_2, train_losses_so_3], axis=0)
# avg_val_losses_so = np.mean([val_losses_so_1, val_losses_so_2, val_losses_so_3], axis=0)
#
# # Calculate average losses for FO
# avg_train_losses_fo = np.mean([train_losses_fo_1, train_losses_fo_2, train_losses_fo_3], axis=0)
# avg_val_losses_fo = np.mean([val_losses_fo_1, val_losses_fo_2, val_losses_fo_3], axis=0)
#
# # Create an array representing the training steps for SO
# train_steps_so = np.arange(len(avg_train_losses_so))
# val_steps_so = np.arange(len(avg_train_losses_so), step=500)
#
# # Create an array representing the training steps for FO (assuming it has the same length as SO)
# train_steps_fo = np.arange(len(avg_train_losses_fo))
#
# # Plotting the data
#
# plt.plot(train_steps_so, avg_train_losses_so, label='SO-Training Loss', color='blue')
# plt.plot(train_steps_fo, avg_train_losses_fo, label='FO-Training Loss', color='grey')
# plt.plot(val_steps_so, avg_val_losses_so.mean(axis=1), label='SO-Validation Loss', color='orange')
# plt.plot(val_steps_so, avg_val_losses_fo.mean(axis=1), label='FO-Validation Loss', color='red')
#
# # Adding labels and title
# plt.xlabel('Meta-Training Steps')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Time')
#
# # Adding a legend
# plt.legend()
#
# # Display the plot
# plt.show()
#
#
#
# # Plot for empirical question 3 second order
# def moving_average(data, window_size):
#     weights = np.repeat(1.0, window_size) / window_size
#     return np.convolve(data, weights, 'valid')
#
# # Load the training and validation loss data for SO
# SO_test_acc5_run1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN1/test-accuracy1.npy")
# SO_test_acc5_run2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN2/test-accuracy2.npy")
# SO_test_acc5_run3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 SO-RUN3/test-accuracy3.npy")
#
# SO_test_acc10 = np.load("/content/drive/MyDrive/AML Results/Classes per task 10 SO-RUN 1/test-accuracy.npy")
# SO_test_acc20 = np.load("/content/drive/MyDrive/AML Results/Classes per task 20 SO-RUN 1/test-accuracy.npy")
#
#
# SO_test_acc5 = np.mean([SO_test_acc5_run1, SO_test_acc5_run2, SO_test_acc5_run3], axis=0)
#
#
# # Apply moving average to SO data
# window_size = 30  # Adjust this as needed
# SO_test_acc5_ma = moving_average(SO_test_acc5, window_size)
# SO_test_acc10_ma = moving_average(SO_test_acc10, window_size)
# SO_test_acc20_ma = moving_average(SO_test_acc20, window_size)
#
# # Plotting the smoothed SO data
# plt.plot(SO_test_acc5_ma, label='SO-5 Classes')
# plt.plot(SO_test_acc10_ma, label='SO-10 Classes')
# plt.plot(SO_test_acc20_ma, label='SO-20 Classes')
#
# # Adding labels and title
# plt.xlabel('Number of Evaluated Tasks')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy Over Time(Second Order)')
#
# # Adding a legend
# plt.legend(loc='upper right')
#
# plt.grid()
# # Display the plot
# plt.show()
#
#
# #Plot for empirical question 3 first order
# def moving_average(data, window_size):
#     weights = np.repeat(1.0, window_size) / window_size
#     return np.convolve(data, weights, 'valid')
#
# # Load the training and validation loss data for FO
# FO_test_acc5_run1 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run1/firstOrderT1_run1_testAcc.npy")
# FO_test_acc5_run2 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run2/firstOrderT1_run2_testAcc.npy")
# FO_test_acc5_run3 = np.load("/content/drive/MyDrive/AML Results/Classes per task 5 FO-RUN 1,2,3/firstOrderT1_run3/firstOrderT1_run3_testAcc.npy")
#
# FO_test_acc10 = np.load("/content/drive/MyDrive/AML Results/Classes per task 10 FO-RUN1/test-accuracy.npy")
# FO_test_acc20 = np.load("/content/drive/MyDrive/AML Results/Classes per task 20 FO-RUN1/test-accuracy.npy")
#
# FO_test_acc5 = np.mean([FO_test_acc5_run1, FO_test_acc5_run2, FO_test_acc5_run3], axis=0)
#
# # Apply moving average to FO data
# FO_test_acc5_ma = moving_average(FO_test_acc5, window_size)
# FO_test_acc10_ma = moving_average(FO_test_acc10, window_size)
# FO_test_acc20_ma = moving_average(FO_test_acc20, window_size)
#
# # Plotting the smoothed FO data
# plt.plot(FO_test_acc5_ma, label='FO-5 Classes')
# plt.plot(FO_test_acc10_ma, label='FO-10 Classes')
# plt.plot(FO_test_acc20_ma, label='FO-20 Classes')
#
# # Adding labels and title
# plt.xlabel('Number of Evaluated Tasks')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy Over Time')
#
# # Adding a legend
# plt.legend(loc='upper right')
#
# plt.grid()
# # Display the plot
# plt.show()