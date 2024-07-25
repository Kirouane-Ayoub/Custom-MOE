# Building a Custom Mixture of Experts Model

This repository contains the complete codebase for building and training a Mixture of Experts (MoE) model from scratch [Project guide](https://blog.gopenai.com/building-a-custom-mixture-of-experts-model-for-our-darija-from-tokenization-to-text-generation-b30ba49ae7cc). 



### Run Trainer 

To use the script with command-line arguments, you can run it from the terminal like this:

```bash
python trainer.py --batch_size 8 --save_path "my_model.pt" --total_steps 15000 --eval_interval 2000
```

Here’s a breakdown of the arguments:

- `--batch_size 8`: Sets the batch size for both training and evaluation to 8.
- `--save_path "my_model.pt"`: Specifies the file path where the model checkpoint will be saved.
- `--total_steps 15000`: Sets the total number of training steps to 15,000.
- `--eval_interval 2000`: Sets the interval for evaluation and logging to every 2,000 steps.

You can also omit any of these arguments to use the default values:

```bash
python trainer.py
```

In this case, the script will use the following defaults:

- Batch size: 4
- Save path: `MOE_darija.pt`
- Total steps: 10,000
- Evaluation interval: 1,000



##  Run Inference 

**Run the script with the command-line arguments:**

```bash
python inference.py --max_length 50 --num_return_sequences 5 --input_text "راك" --model_path "path/your_model.pt"
```

**Explanation of Arguments**

- `--max_length 50`: Sets the maximum length of the generated sequences to 50 tokens.
- `--num_return_sequences 5`: Specifies the number of sequences to generate as 5.
- `--input_text "راك"`: Provides the input text for generation. Default is `"راك "`.
- `--model_path "path/your_model.pt"`: Path to the trained model checkpoint file. This is required to load the model.

If you don’t specify `--input_text`, the default value `"راك "` will be used. If you don’t specify `--max_length` or `--num_return_sequences`, the default values of 30 and 10 will be used, respectively.
