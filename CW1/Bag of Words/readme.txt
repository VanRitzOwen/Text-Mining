Readme for model_1
############# NOTE#############

Because this BOW model is implemented on jupyter, different steps implemented in different cells
Considering that if you run the entire program at once, there will be too much content to printed, 
So in order to make the output more clear, I added '#' before some print functions. 
The entire program now will only output the accuracy rate before training and the accuracy after training. 
If you want to see some result, just follow the instruction and remove the "#"

#################################



1. This model need these files : train_data.txt , test.txt, glove.small.txt, allTag.txt.
So put bow_model_1.ipynb and these four txt files in one folder.



2. Run the code
If you run the entire program at once, you see two results, 
One is the accuracy of the untrained model, and the other is the accuracy of the trained model.



If you want to see some details in the model, you can just remove some '#' before the print function.
They are:
1. #print(data[0]) : 
You can see the format of the training data after extracted from the file and preprocessed.

2.#print(VOCAB_SIZE)
How many words in the dictionary.

3.#print(getWordvector.get('donate'))
It is a example , you can input a word, if this word is in the glove file it will return its word vector.

4. #print(instance)
    #print("The predict label is :" + ix_to_label[index_max]+"   The real label is "+ label)

This step can help you clearly see the sentence and the result of prediction, also the correct label of this instance.



