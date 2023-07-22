from lippy.utils.speaker import Speaker
from lippy.utils.listener import Listener
from os import listdir
# from thefuzz import fuzz
from rouge_score import rouge_scorer
from nltk import sent_tokenize,download


if __name__ == "__main__":
    promptGroundTruth = "1. Gas chromatography is a separation technique that separates compounds in a gaseous mixture based on their physical properties. It is used in the field of chemical analysis to separate and identify various elements present in a sample. The process involves passing a gas through a small column containing a stationary phase. The stationary phase is usually packed into the column to separate the molecules. The molecules then interact with each other and the stationary phase, which leads to a separation and identification of individual components in the mixture."

    nameWav = "proof/genOP"
    pathWav = "/home/ubuntu/Tehas/lippy/data/audio/" + nameWav

    speaker = Speaker(speaker_id = "Dalinar-1_t8_w8_7")
    speaker.say(promptGroundTruth, nameWav)

    listener = Listener()
    promptTranscript = listener.transcribe(pathWav + ".wav")
    print(f"Ground Truth: {promptGroundTruth}")
    print("---")
    print(f"Transcription: {promptTranscript['text']}")

    download('punkt')
    # tokenGroundTruth = sent_tokenize(promptGroundTruth)
    # tokenTranscript = sent_tokenize(promptTranscript['text'])
    # tokens = zip(tokenGroundTruth, tokenTranscript)
    # for hyps, refs in list(tokens):
    #     print(f"""
    #     GTs: {hyps}
    #     TRs: {refs}
    #     ---
    #     """)
    rogue = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    print(rogue.score(promptGroundTruth, promptTranscript['text']))