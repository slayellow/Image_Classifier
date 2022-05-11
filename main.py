from classifier.image_classifier import ImageClassifier


if __name__ == '__main__':
    classifier = ImageClassifier()
    classifier.load_dataset()
    classifier.set_dataset_detail()
    classifier.set_model()
    classifier.set_loss()
    classifier.set_optimizer()
    classifier.train()
    classifier.visualization_loss_graph()
    classifier.finish()
