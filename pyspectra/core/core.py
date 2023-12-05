import numpy as np
from pyspectra.exception import pyspectra_exception

class PySpectra(object):
    def __init__(self):
        pass

    def _lsbEmbed(self, image: np.ndarray, message: str) -> np.ndarray:
        """
        Embeds a message into the least significant bit of each pixel in a grayscale image.
        
        Args:
        - image: Grayscale image as a numpy array
        - message: Message to be embedded
        
        Returns:
        - Stego image with embedded message

        Embeds the given message into the least significant bit (LSB) of each pixel in the provided 
        grayscale image. Before embedding, it checks if the image is grayscale and validates if the 
        message size is within the capacity of the image. The message is embedded bit by bit into the 
        LSB of each pixel by modifying their values.
        """

        if len(image.shape) != 2:  # Check if the image is grayscale
            raise ValueError("Input image should be a grayscale image")

        # Check if the message can fit within the image
        max_message_size = image.size 
        if len(message) > max_message_size:
            raise ValueError("Message size exceeds the capacity of the image")

        # Flatten the image to process pixel values
        flat_image = image.flatten()

        # Embedding message in the least significant bit of each pixel
        bit_index = 0
        for char in message:
            flat_image[bit_index] = (flat_image[bit_index] & 0xFE) | int(char)
            bit_index += 1

        # Reshape the modified array back to the original image shape
        stego_image = flat_image.reshape(image.shape)
        return stego_image


    def _getLSB(self, image: np.ndarray) -> str:
        """
        Retrieves the least significant bit of each pixel in an image.
        
        Args:
        - image: Image as a numpy array
        
        Returns:
        - Binary string of least significant bits

        Retrieves the least significant bit (LSB) of each pixel in the provided image by flattening 
        the image and extracting the LSB of each pixel value. The resulting binary string represents 
        the collection of LSBs for all pixels in the image.
        """

        # Flattens the image to process pixel values
        flat_image = image.flatten()
        lsbMessage = ""

        # Iterates through each pixel and extracts its LSB
        for i in range(len(flat_image)):
            lsbMessage += str(flat_image[i] & 1) # shortcut to get last bit
        return lsbMessage


    def _spreading(self, message: str, scalar: int) -> str:
        """
        Spreads a binary message by duplicating each bit by a scalar value.
        
        Args:
        - message: Binary message to be spread
        - scalar: Scalar value for spreading
        
        Returns:
        - Spreaded binary message

        Converts the input binary message into a binary string representation. Then, for each bit in 
        the binary message, duplicates the bit 'scalar' times, creating a spreaded binary message where each original bit is repeated 'scalar' times.
        """

        # scalar value must be greather than zero
        if (scalar < 1):
            raise pyspectra_exception.PySpectraException(message="Scalar value must be greather than zero")

        # Converts the input message to binary representation
        message_bin = "".join(format(ord(i), '08b') for i in message)
        spreaded_message = ""
        
        # Duplicates each bit in the message by the given scalar value
        for i in message_bin:
            spreaded_message += i * scalar
            
        return spreaded_message
    

    def _deSpreading(self, message: str, scalar: int) -> str:
        """
        De-spreads a binary message by extracting original bits from spreaded pairs.
        
        Args:
        - message: Binary message to be de-spread
        - scalar: Scalar value for de-spreading
        
        Returns:
        - De-spreaded binary message

        De-spreads the given binary message by extracting the original bits from spreaded pairs. 
        It iterates through the message in groups of 'scalar' length, checking if all bits in each 
        group are identical. If so, it appends that bit to the de-spreaded message; otherwise, 
        it raises an error indicating a broken binary message.
        """

        original_message = ""

        # Iterates through the message in groups of 'scalar' length
        for i in range(0, len(message), scalar):
            pair = message[i:i+scalar]

            # Checks if all bits in the group are identical
            if all(bit == pair[0] for bit in pair):
                original_message += pair[0] # Appends the bit to the de-spreaded message
            else:
                raise pyspectra_exception.PySpectraException(message="Binary is broken, please check the key again")
            
        return original_message
    

    def _generateSeed(self, key: str) -> int:
        """
        Generates a seed for pseudo-random number generation based on the key.
        
        Args:
        - key: Input key for seed generation
        
        Returns:
        - Generated seed as a string

        Generates a seed for pseudo-random number generation by performing an XOR operation on the 
        binary representation of characters in the given key. It converts each character of the key 
        into its 8-bit binary representation and iteratively performs XOR operations between these 
        binary values to derive the final seed as a string.
        """

        # Converts each character of the key into its 8-bit binary representation
        key_binary = [format(ord(i), '08b') for i in key]
        bin_left = key_binary[0]
        
        # Performs XOR operations on binary representations of characters in the key
        for bin_right in key_binary[1:]:
            int_left = int(bin_left, 2)
            int_right = int(bin_right, 2)
            
            # Performs XOR operation between binary representations
            xor_result = int_left ^ int_right
            
            # Converts the result back to binary format
            bin_result = bin(xor_result)[2:].zfill(8)
            
            bin_left = bin_result
            
        return int(bin_left, 2)


    def _generatePseudoNoise(self, seed: int, length: int) -> str:
        """
        Generates pseudo-random noise based on a seed using a Linear Congruential Generator (LCG).
        
        Args:
        - seed: Seed value for noise generation
        - length: Length of the generated noise
        
        Returns:
        - Generated pseudo-random noise as a string

        Generates pseudo-random noise using a Linear Congruential Generator (LCG) with the specified seed. The LCG formula used is Xn+1 = (a * Xn + c) % m, where 'a', 'c', and 'm' are constants. 
        For the provided example, 'a' is 7, 'c' is 1, and 'm' is 32. It generates 'length' amount of 
        pseudo-random numbers by iteratively applying the LCG formula and converting each result to an 
        8-bit binary string to form the pseudo-random noise string.
        """

        # formula LCG -> Xn+1 = (a . Xn + c) mod m
        # let's say a = 7, c = 1, m = 32
        
        # Constants for the Linear Congruential Generator (LCG)
        a = 7
        c = 1
        m = 32
        result = ""
        current_x = seed
        
        # Generates 'length' amount of pseudo-random numbers using LCG
        for _ in range(length):
            x_next = (a*current_x + c) % m
            result += bin(x_next)[2:].zfill(8) # Converts each number to an 8-bit binary string
            current_x = x_next
        
        return result        


    def _modulation(self, message:str, pseudo_noise: str) -> str:
        """
        Modulates a message with pseudo-random noise using XOR operation.
        
        Args:
        - message: Binary message to be modulated
        - pseudo_noise: Pseudo-random noise for modulation
        
        Returns:
        - Modulated binary message
        
        Modulates the given binary message with the provided pseudo-random noise using the XOR operation. 
        It first verifies that the lengths of the message and pseudo-random noise are equal. Then, it performs 
        an XOR operation between corresponding bits of the message and the pseudo-random noise, generating 
        a modulated binary message where each bit is the result of the XOR operation between the respective 
        bits of the message and noise.
        """

        # Verifies if the lengths of both inputs are the same
        if len(message) != len(pseudo_noise):
            raise ValueError("Length of both input must be the same")

        # Performs XOR operation between each corresponding bit of the message and pseudo-random noise
        modulated_result = ""
        for i in range(len(message)):
            # Retrieves bits from the same position in both inputs
            bit_message = message[i]
            bit_pseudo_number = pseudo_noise[i]

            # XOR operation between bits
            result_bit = '1' if bit_message != bit_pseudo_number else '0'
            modulated_result += result_bit

        return modulated_result



    def _spreadSpectrumEmbed(self, message: str, key: str, scalar: int) -> str:
        """
        Embeds a message using Spread Spectrum technique.
        
        Args:
        - message: Message to be embedded
        - key: Key for spreading
        - scalar: Scalar value for spreading
        
        Returns:
        - Modulated message after spreading and modulation
        
        Embeds the provided message using Spread Spectrum technique, involving spreading the message, 
        generating pseudo-random noise, and modulating the spreaded message with the generated noise. 
        First, the message is spread using the specified scalar value. Then, a pseudo-random noise 
        sequence is generated based on the given key. The length of the generated noise matches the 
        length of the spreaded message, ensuring compatibility for modulation. The spreaded message 
        is modulated by XOR operation with the generated pseudo-random noise, resulting in the modulated message after spreading and modulation.
        """

        # Spread the message with the specified scalar value
        spreaded_message = self._spreading(message=message, scalar=scalar)

        # Generate pseudo-random noise sequence
        seed = self._generateSeed(key)
        inc = 1
        pseudo_number = self._generatePseudoNoise(seed=seed, length=inc*scalar)

        # Ensure the length of pseudo-random noise matches the spreaded message length
        while len(pseudo_number) < len(spreaded_message):
            inc += 1
            pseudo_number = self._generatePseudoNoise(seed=seed, length=inc*scalar)

        # Modulate the spreaded message with the generated pseudo-random noise
        modulated_message = self._modulation(spreaded_message, pseudo_number)

        return modulated_message



    def _spreadSpectrumExtract(self, modulated_message: str, key: str, scalar: int) -> str:
        """
        Extracts an embedded message using Spread Spectrum technique.
        
        Args:
        - modulated_message: Modulated message with embedded data
        - key: Key for extraction
        - scalar: Scalar value for extraction
        
        Returns:
        - Extracted original message
        
        Extracts an embedded message using the Spread Spectrum technique. It begins by generating a 
        pseudo-random noise signal based on the provided key. Next, it demodulates the modulated 
        message by applying the previously generated pseudo-random noise. Then, it performs de-spreading by identifying original bits from spreaded pairs using the specified scalar value. Finally, 
        it returns the extracted original message that was embedded using Spread Spectrum.
        """

        # Generate pseudo-random noise signal based on the provided key
        pseudo_number = self._generatePseudoNoise(seed=self._generateSeed(key), length=len(modulated_message)//8)
        
        # Demodulate the modulated message using the generated pseudo-random noise
        spreaded_message = self._modulation(message=modulated_message, pseudo_noise=pseudo_number)
        
        # De-spread the demodulated message to extract the original message
        original_message = self._deSpreading(message=spreaded_message, scalar=scalar)

        return original_message

    

    def _fromBinToText(self, bin_string: str) -> str:
        """
        Converts a binary string to its equivalent text representation.
        
        Args:
        - bin_string: Binary string to be converted
        
        Returns:
        - Converted text

        Converts the given binary string into its equivalent text representation. It first segments 
        the binary string into groups of 8 bits each, representing individual ASCII characters. 
        Then, it iterates through these segments, converting each 8-bit binary sequence to its 
        corresponding ASCII character using the 'int(binary, 2)' function, and finally joins 
        these characters together to form the resulting text.
        """

        # Split the binary string into segments of 8 bits each
        binary_list = [bin_string[i:i+8] for i in range(0, len(bin_string), 8)]

        # Convert each binary segment to its corresponding ASCII character and join them
        text = ''.join(chr(int(binary, 2)) for binary in binary_list)

        return text
    

    def _getPattern(self, lsbData: str) -> tuple[str, str, str]:
        """
        Extracts specific patterns from LSB data.
        
        Args:
        - lsbData: LSB data containing embedded patterns
        
        Returns:
        - Extracted patterns from LSB data
        
        Extracts specific patterns from the provided LSB (Least Significant Bit) data. The method 
        slices the input LSB data to extract different patterns. It retrieves the first 64 characters 
        as the length of the modulated message, the next 8 characters as the scalar value used in 
        spreading, and the remaining characters as the modulated message with embedded data.
        """

        # Extract specific patterns from the LSB data
        length_modulated_message = lsbData[:64]  # Extracts the first 64 characters as message length
        scalar_value = lsbData[64:64+8]  # Extracts the next 8 characters as scalar value
        embedded_message = lsbData[64+8:]  # Extracts the rest as the modulated message with embedded data

        return length_modulated_message, scalar_value, embedded_message

    

    def embedding(self, image: np.ndarray, message: str, key: str, scalar: int) -> np.ndarray:
        """
        Embeds a message into an image using Spread Spectrum and LSB techniques.
        
        Args:
        - image: Image to embed the message
        - message: Message to be embedded
        - key: Key for embedding
        - scalar: Scalar value for embedding
        
        Returns:
        - Image with embedded message
        
        Embeds the provided message into the given image using a combination of Spread Spectrum 
        and LSB (Least Significant Bit) techniques. First, it employs Spread Spectrum by modulating 
        the message to create a modulated message. Then, it generates a binary pattern representing 
        the length of the modulated message and the scalar value. This pattern is concatenated with 
        the modulated message to form the final message to be embedded. The LSB technique is then used 
        to embed this final message into the provided image by modifying the least significant bit 
        of each pixel. Finally, it returns the image with the embedded message.
        """

        # Modulate the message using Spread Spectrum
        modulated_message = self._spreadSpectrumEmbed(message=message, key=key, scalar=scalar)

        # Generate a binary pattern representing the length of the modulated message and scalar value
        length_modulated_message = len(modulated_message)
        bin_len_modulated = bin(length_modulated_message)[2:].zfill(64)
        bin_len_scalar = bin(scalar)[2:].zfill(8)

        # Combine the binary patterns with the modulated message
        message = bin_len_modulated + bin_len_scalar + modulated_message

        # Embed the final message into the image using LSB
        embedded_image = self._lsbEmbed(image=image, message=message)

        return embedded_image
    
    def extract(self, image: np.ndarray, key: str) -> str:
        """
        Extracts a message from an image using Spread Spectrum and LSB extraction methods.
        
        Args:
        - image: Image from which to extract the message
        - key: Key for message extraction
        
        Returns:
        - Extracted message
        
        Extracts a message from the provided image using a combination of Spread Spectrum and LSB 
        extraction methods. Initially, it retrieves the least significant bit (LSB) data from the 
        image, which contains the embedded patterns. It then extracts specific patterns from the LSB 
        data to obtain information about the length of the modulated message and the scalar value used 
        in embedding. With this information, it extracts the modulated message containing the embedded 
        data. Finally, it utilizes the Spread Spectrum extraction method to retrieve the original 
        message by reversing the embedding process using the provided key. The extracted original 
        message is then converted from binary to text format before being returned.
        """

        # Retrieve the LSB data from the image
        lsbData = self._getLSB(image=image)

        # Extract patterns (message length, scalar value, and modulated message) from LSB data
        len_message_bin, scalarBin, message = self._getPattern(lsbData=lsbData)
        len_message = int(len_message_bin, 2)
        scalar = int(scalarBin, 2)

        # Extract the original message using Spread Spectrum extraction method
        original_message = self._spreadSpectrumExtract(modulated_message=message[:len_message], key=key, scalar=scalar)

        # Convert the extracted original message from binary to text format
        return self._fromBinToText(original_message)
