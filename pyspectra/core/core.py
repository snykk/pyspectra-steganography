import numpy as np

class PySpectra(object):
    def __init__(self):
        pass

    def _lsbEmbed(self, image=np.array([]), modulated_message="") -> np.array([]):
        if len(image.shape) != 2:  # Check if the image is grayscale
            raise ValueError("Input image should be a grayscale image")

        # Check if the message can fit within the image
        max_message_size = image.size 
        if len(modulated_message) > max_message_size:
            raise ValueError("Message size exceeds the capacity of the image")

        # Flatten the image to process pixel values
        flat_image = image.flatten()

        # Embedding message in the least significant bit of each pixel
        bit_index = 0
        for char in modulated_message:
            flat_image[bit_index] = (flat_image[bit_index] & 0xFE) | int(char)
            bit_index += 1

        # Reshape the modified array back to the original image shape
        stego_image = flat_image.reshape(image.shape)
        return stego_image

    def _getLSB(self, image=np.array([])) -> np.array([]):
        flat_image = image.flatten()
        lsbMessage = ""
        for i in range(len(flat_image)):
            lsbMessage += str(flat_image[i] & 1) # shortcut to get last bit
        return lsbMessage

    def _spreading(self, message="", scalar=2) -> str:
        message_bin = "".join(format(ord(i), '08b') for i in message)
        spreaded_message = ""
        
        for i in message_bin:
            spreaded_message += i*scalar
            
        return spreaded_message

    def _deSpreading(self, message="", scalar=2) -> str:
        original_message = ""

        for i in range(0, len(message), scalar):
            original_message += message[i]
            
        return original_message
    
    def _generateSeed(self, key="") -> str:
        key_binary = [format(ord(i), '08b') for i in key]
        bin_left = key_binary[0]
        
        for bin_right in key_binary[1:]:
            int_left = int(bin_left, 2)
            int_right = int(bin_right, 2)
            
            xor_result = int_left ^ int_right
            
            bin_result = bin(xor_result)[2:].zfill(8)
            
            bin_left = bin_result
            
        return int(bin_left, 2)

    def _generatePseudoNoise(self, seed=0, length=2) -> str:
        # formula LCG -> Xn+1 = (a . Xn + c) mod m
        # let's say a = 27, c = 7, m = 21
        
        a = 7
        c = 1
        m = 32
        result = ""
        current_x = seed
        
        for _ in range(length):
            x_next = (a*current_x + c) % m
            result += bin(x_next)[2:].zfill(8)
            current_x = x_next
        
        return result        

    def _modulation(self, message="", pseudo_noise="") -> str:
        pass
        # modulate message with pseudo noise using XOR operation
        # ensure the length is same
        if len(message) != len(pseudo_noise):
            raise ValueError("Length of both input must be the same")

        # do xor operation each bit
        modulated_result = ""
        for i in range(len(message)):
            # get bit of the binary in the same position
            bit_message = message[i]
            bit_pseudonumber = pseudo_noise[i]

            # xor
            result_bit = '1' if bit_message != bit_pseudonumber else '0'
            modulated_result += result_bit

        return modulated_result

    def _spreadSpectrumEmbed(self, message="", key="", scalar=2) -> str:
        pass
        # 1. spread message with scalar value
        # 2. generate pseudo noise with length of message that have been spreaded
        # 3. modulate that message with pseudo noise that have been generate

        spreaded_message= self._spreading(message=message, scalar=scalar)

        # generate pseudo number
        inc = 1
        pseudo_number = self._generatePseudoNoise(self._generateSeed(key), inc*scalar)
        while len(pseudo_number) < len(spreaded_message):
            inc += 1
            pseudo_number = self._generatePseudoNoise(self._generateSeed(key), inc*scalar)

        # modulate message
        modulated_message = self._modulation(spreaded_message, pseudo_number)

        return modulated_message



    def _spreadSpectrumExtract(self, modulated_message="", key="", scalar=2) -> str:
        pass
        # 1. generate pseudo noise signal
        # 2. demoduate
        # 3. de spreading
        # 4. get the extracted message

        pseudo_number = self._generatePseudoNoise(self._generateSeed(key), len(modulated_message)//8)
        spreaded_message = self._modulation(message=modulated_message, pseudo_noise=pseudo_number)
        original_message = self._deSpreading(message=spreaded_message, scalar=scalar)

        return original_message

    def embedding(self, image=np.array([]), message="", key="", scalar=2) -> np.array([]):
        pass
        # 1. modulate message with Spread Spectrum Core
        # 2. embed message in to the image LSB Core
        modulated_message = self._spreadSpectrumEmbed(message=message, key=key, scalar=scalar)
        print("[embedding] modulated message:", modulated_message)
        embedded_image = self._lsbEmbed(image=image, modulated_message=modulated_message)

        return embedded_image
    
    def _fromBinToText(self, bin_string="") -> str:
        binary_list = [bin_string[i:i+8] for i in range(0, len(bin_string), 8)]

        # Convert binary segments to ASCII characters and join them
        text = ''.join(chr(int(binary, 2)) for binary in binary_list)

        return text

    def extract(self, image=np.array([]), key="", scalar=2):
        pass
        # 1. Get the last bit of an image
        # 2. extract message with spread spectrum extract
        lsbMessage = self._getLSB(image=image)[:128]
        original_message = self._spreadSpectrumExtract(modulated_message=lsbMessage, key=key, scalar=scalar)

        return self._fromBinToText(original_message)
    
        