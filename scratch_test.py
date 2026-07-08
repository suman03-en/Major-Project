from pdf_extractor import extract_metadata_from_text
from dataset.pipeline.formatter import extract_references

front_matter = """
कम्पनी ऐन, २०६३ 
प्रमाणीकरण मिति  
२०६३।0७।२४ 
संशोधन गर्ने ऐन 
१. केही नेपाल ऐनलाई संशोधन गर्ने ऐन, २०६४    २०६४।0५।0९ 
२. कम्पनी (पहिलो संशोधन) ऐन, २०७४            २०७४।0१।१९ 
परिच्छेद-२ 
"""

metadata = extract_metadata_from_text(front_matter)
print("Metadata:")
print(metadata)

text_with_refs = "उपदफा (१) मा जुनसुकै कुरा लेखिएको भए तापनि दफा २३ बमोजिमको खण्ड (क) लागु हुनेछ।"
refs = extract_references(text_with_refs, "company-act")
print("\nRefs:")
print(refs)
