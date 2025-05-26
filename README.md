# Expansion on Zain’s Work

See Zain’s README for details on the underlying model.

## The Problem

Zain’s model works very effectively when a generated image is given to the decoder. However, the real world application involves sending the generated image through some medium. Maybe a group chat or social platform. These applications often compress images to save on cost.

I tested the effectiveness of Zain’s model when compressed by applying JPEG compression to the images created by the generator before passing them to the decoder. The results were only a little better than randomly guessing bits.

## The Idea

Make changes to the training process to improve resistance to compression. Before the decoder attempts to extract a message during training, the image will be compressed. This way, the decoder will find new ways to improve its performance specifically when dealing with compressed images.

## Implementation Details

See changes to `DCGAN_based_trainer_v2_4.py`. A QF of 75 was used to compress the images. Model was trained for 30 epochs.

See `compressiontest.py` and `non-compressiontest.py`, plus code contributions below, to understand how each model's performance was tested.

## Testing Results

| Generated Image Quality  | Old Model Bits Matched | New Model Bits Matched |
|--------------------------|------------------------|-------------------------|
| No compression           | 100%                   | 83%                     |
| JPEG compression QF = 75 | 59%                    | 83%                     |
| JPEG compression QF = 30 | 58%                    | 80%                     |

## Result Thoughts

80% is ok but not good enough as is. Could be improved with more training, or mitigated with error correction.

## My Code Contributions

- Rewrote image generation and decoding scripts by breaking them up into functions. Allows me to import them into other files, and enables the below test scripts.

- Added non-compression test script, used to measure performance of models in a perfect channel with no compression.
  - creates a random message  
  - using the generator, creates an image  
  - using the decoder, attempts to extract the message  
  - gives a % of matching bits between original and extracted message  
  - repeats multiple times and gives an average  

- Added compression test script, just like above, but the generated image is compressed before being passed to the decoder.

- Create a new DCGAN trainer that trains on generated images that have gone through compression.

- The resulting generator, decoder, and discriminators created by the new trainer
