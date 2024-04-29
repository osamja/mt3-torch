# TODO
* 


# IN PROGRESS
* Understand the Current Implementation

# DONE


# ChatGPT Prompt
I would like to rewrite the mt3 library to run on pure pytorch.  This repo has so many dependencies that it is quite hard to build on other computers.  I would like to simplify the build process by using less libraries, and learning more about the implementation by reimplementing this library in pytorch.  Can you please help me?

https://github.com/magenta/mt3

That sounds like an exciting and ambitious project! Rewriting the MT3 library (part of Google's Magenta project) to run purely on PyTorch will definitely help in understanding its internals and might make the library more accessible for some users.

Here are some steps and considerations to guide you through the process of rewriting the MT3 library in PyTorch:

    Understand the Current Implementation:
        Familiarize yourself with the existing codebase of MT3. Understand the high-level components like data preprocessing, model architecture, training routines, and the inference pipeline.
        Identify the core dependencies that are critical for the functionality (like TensorFlow, TensorFlow Datasets, etc.) and think about their PyTorch equivalents.

    Plan the PyTorch Architecture:
        Map out the components of MT3 that you need to reimplement in PyTorch. This includes the neural network models, any custom layers or functions, data loaders, etc.
        Decide on how to handle datasets. MT3 might be using specific formats or pipelines for feeding data into the model. PyTorch has torch.utils.data.Dataset and torch.utils.data.DataLoader which can be utilized here.

    Model Conversion:
        Start with the neural network architecture. Convert the TensorFlow model definitions to PyTorch by rewriting the layers, activations, and connections.
        Ensure that model outputs and shapes remain consistent with the original implementation to verify correctness.

    Data Handling and Processing:
        Adapt data preprocessing code to work with PyTorch. This includes any normalization, augmentation, and dataset manipulation.
        Implement custom dataset classes if required, especially if the data involves complex structures or preprocessing.

    Training and Evaluation:
        Rewrite the training loops. This includes handling the forward pass, loss computation, backpropagation, and parameter updates.
        Implement evaluation metrics and validation checks to compare the performance of your PyTorch version against the original TensorFlow version.

    Utilities and Miscellaneous Components:
        Address additional utilities like saving/loading models, logging, and configuration handling, which might be tightly coupled with TensorFlow or other libraries in the current implementation.

    Testing and Debugging:
        Thoroughly test each component separately and the system as a whole. Ensure that the model’s performance metrics are on par with the original implementation.
        It's beneficial to write unit tests for critical components to ensure stability and reproducibility.

    Documentation and Examples:
        Update or write new documentation to reflect changes in the codebase and setup instructions.
        Provide example scripts demonstrating how to use the re-written library.

    Community Feedback and Contributions:
        Consider hosting your version on GitHub and encourage the community to contribute and provide feedback. Open sourcing your work can lead to improvements and new ideas.