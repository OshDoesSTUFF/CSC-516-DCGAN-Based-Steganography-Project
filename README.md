# Expansion on Zain’s Work

See Zain’s README for details on the underlying model.

## The Problem

Zain’s model works very effectively when a generated image is given to the decoder. However, the real world application involves sending the generated image through some medium. Maybe a group chat or social platform. These applications often compress images to save on cost.

I tested the effectiveness of Zain’s model when compressed by applying JPEG compression to the images created by the generator before passing them to the decoder. The results were only a little better than randomly guessing bits.

## Testing Methodology

- A random string of 12 ascii characters is generated.

  `abcdefghijkl`

- The string is fed to the generator function, which zero pads the message to 100 bytes (the maximum capacity of our implementation)

  `01100001 01100010 01100011 01100100 01100101 01100110 01100111 01101000 01101001 01101010 01101011 01101100`

- The generator function creates an original image, using the 100 bytes as a noise vector

    ![new_stego_image](https://github.com/user-attachments/assets/f800d802-95f2-49c6-9201-15e2f0592b01)

- The generated image is compressed using JPEG compression. The quality of compression is determined by the “QF” parameter. The lower the QF, the greater the compression of the image, and the lower the perceived quality. 

    ![new_stego_image_q30](https://github.com/user-attachments/assets/00d7a9f8-1557-41f1-a3ee-2a74f1565936)

- The image, now compressed, is fed into the decoder function. The decoder attempts to extract the original string from the image.

  `ddcbezgwilkv`

- The original generated string is compared to the extracted string that came from the decoder. The comparison is bitwise, not by character. A % match is given.

  ```
  01100001 01100010 01100011 01100100 01100101 01100110 01100111 01101000 01101001 01101010 01101011 01101100
  01100100 01100100 01100011 01100010 01100101 01111010 01100111 01110111 01101001 01101100 01101011 01110110
  ```

- The above is repeated 200 times and an average % match is given.

## Initial Testing Results

| Generated Image Quality  |  Bits Matched
|--------------------------|------------------------|
| No compression           | 100%                   | 
| JPEG compression QF = 75 | 59%                    | 
| JPEG compression QF = 30 | 58%                    | 

## The Idea

Make changes to the training process to improve resistance to compression. Before the decoder attempts to extract a message during training, the image will be compressed. This way, the decoder will find new ways to improve its performance specifically when dealing with compressed images.

## Implementation Details

During training, the generated image is compressed before the decoder attempts to extract a message. A QF of 75 was used to compress the images. The model was trained for 30 epochs. The resulting model was put through the same circuit of tests as the original model.

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
