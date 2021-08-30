import tensorflow as tf
import dataget
import model as md
import predictions as pred

train_images, train_labels, test_images, test_labels, class_names = dataget.dataget()
model = md.defmodel()

model = md.compilemodel(model)

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest loss: ", test_loss, "\nTest accuracy: ", test_acc)

predictions, model = pred.probmodel(model, test_images)
pred.plot_preds(predictions)

model.save('My_Model1.h5')

model.summary()
