import os
import torch
from scipy.io import wavfile


class Speaker:
    def __init__(self, base_dir: str = "audio_files"):
        """
        Initializes the Speaker class.

        Parameters:
        base_dir (str, optional): The base directory where the audio files will
                                  be saved.
                                  Defaults to "audio_files".
        """
        self.lang = 'en'
        self.model_id = 'v3_en'
        self.samp = 48000
        self.speaker = 'en_0'
        self.device = torch.device('cpu')
        self.base_dir = base_dir
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained TTS model.
        """
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self.lang,
            speaker=self.model_id
        )
        self.model.to(self.device)
        
    def split_into_sentences(self, input_string):
        # Split the input string into individual sentences
        sentences = input_string.split('. ')

        # Initialize variables
        result = []
        current_sentence = ''

        # Iterate over each sentence
        for sentence in sentences:
            # Append the current sentence and the next sentence
            temp = current_sentence + sentence + '. '

            # Check if the combined sentence is within the limit
            if len(temp) <= 900:
                current_sentence = temp
            else:
                # Split the sentence at word boundaries
                words = sentence.split()
                temp = current_sentence

                for word in words:
                    # Check if adding the word exceeds the limit
                    if len(temp) + len(word) <= 900:
                        temp += word + ' '
                    else:
                        # Add the current sentence to the result
                        result.append(temp)
                        temp = word + ' '

                # Add the remaining sentence to the result
                if temp.strip():
                    result.append(temp)

                current_sentence = ''

        # Add the last sentence to the result
        if current_sentence.strip():
            result.append(current_sentence)

        return result

    def say(self, input:str) -> torch.Tensor:
        strings = self.split_into_sentences(input)
        outputs = []
        for i, string in enumerate(strings):
            outputTensor = self.speak(string)
            outputs.append(outputTensor)
        outputs = torch.cat(outputs)
        self.save_audio(outputs, f'output.wav',True)
            
        return outputs
    
    def speak(self, input: str) -> torch.Tensor:
        """
        Generates speech from the input text.

        Parameters:
        input (str): The text to be spoken.

        Returns:
        torch.Tensor: The generated speech audio as a tensor.
        """
        return self.model.apply_tts(
            text=input,
            speaker=self.speaker,
            sample_rate=self.samp
        )

    def save_audio(
        self,
        audio_output: torch.Tensor,
        filename: str,
        overwrite: bool = False
    ):
        """
        Saves the audio as a WAV file.

        Parameters:
        audio_output (torch.Tensor): The audio to be saved.
        filename (str): The filename for the saved audio.
        overwrite (bool, optional): Whether to overwrite an existing file
                                    with the same name.
                                    Defaults to False.
        """
        path = os.path.join(self.base_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            print(f"File {path} already exists.")
            return

        print(f'Saving {filename} to {self.base_dir}')
        wavfile.write(path, self.samp, audio_output.detach().numpy())

if __name__ == "__main__":
    voice = Speaker("/home/ubuntu/Tehas/lippy/data/audio")
    text = """
1. Gas chromatography is a separation technique that separates compounds in a gaseous mixture based on their physical properties. It is used in the field of chemical analysis to separate and identify various elements present in a sample. The process involves passing a gas through a small column containing a stationary phase. The stationary phase is usually packed into the column to separate the molecules. The molecules then interact with each other and the stationary phase, which leads to a separation and identification of individual components in the mixture.
2. The stationary phase in gas chromatography is a substance that has a high affinity for certain molecules. The stationary phase material needs to be inert, meaning that it will not react with the molecules that are being separated. A common type of stationary phase used in gas chromatography is a solid catalyst. In gas chromatography, the gas and the stationary phase flow through the column at different rates and are combined at the inlet. The gas is usually passed in a carrier gas to maintain a constant flow rate.
3. Gas chromatography has a wide range of applications, from pharmaceutical analysis to forensics. It is used to analyze components in air, water, and soil. The gas chromatography process is used in the analysis of drugs, pesticides, and heavy metals in environmental samples, and to detect impurities in industrial gases. Gas chromatography can also be used to analyze the components of gases produced by combustion or chemical processes.
4. To separate and identify components in a gas mixture, a gas chromatography process must be used. Gas chromatography involves the use of a stationary phase material that interacts with the molecules in a gas mixture. The stationary phase material is packed into the gas column and then activated to form a high-affinity chemical bond between the gas molecules and the stationary phase. The molecules are then separated according to their different affinities, and identified by the specific reaction they undergo with the stationary phase. The result is a mixture of gas components, which can be used for identification and quantification purposes.
5. The main components of a gas chromatography process are the gas source (usually air or nitrogen), column (containing the stationary phase material), and detector. The gas source is typically compressed and introduced into the column to create the gas chromatography flow at a pressure lower than atmospheric pressure. The gas mixture is then introduced into the column to separate and identify components in the mixture.
6. Gas chromatography has several applications in the separation and detection of components in gaseous mixtures, such as chemical analysis of food and beverage, analysis of pharmaceutical products, and environmental analysis. The separation of gas components using gas chromatography is an effective method to achieve a high degree of resolution and accuracy. Gas chromatography is also used in laboratories and industry to detect the presence of impurities in the atmosphere. The identification of compounds present in a gas mixture is achieved through the reaction of these components with the stationary phase material in the column. The detection of compounds using gas chromatography is used to ensure the quality of food and beverages as well as to meet strict environmental regulations.
"""
    voice.say(text)