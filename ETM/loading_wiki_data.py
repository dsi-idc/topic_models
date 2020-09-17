import xml.etree.ElementTree as etree
import numpy as np
import pickle
import os


# configurations
wiki_to_load = 'ar'# 'he' # 'ar'
articles_amount_limit = 200000
saving_path = 'C:\\Users\\avrahami\\Documents\\Private\\IDC\\influencer_influenced_project\\arabic_wiki_dataset'
saving_f_name = 'concise_arabic_wiki_data_2_9_2020.p'

if wiki_to_load == 'ar':
    xml_path = 'C:\\Users\\avrahami\\Documents\\Private\\IDC\\influencer_influenced_project\\arabic_wiki_dataset\\' \
               'arwiki-20180920-corpus.xml'
elif wiki_to_load == 'he':
    xml_path = 'C:\\Users\\avrahami\\Documents\\Private\\IDC\\influencer_influenced_project\\arabic_wiki_dataset\\' \
               'hewiki-20181001-corpus.xml'
else:
    raise IOError("Invalid wiki language source, has to be either 'ar' or 'he'. Please specify")

textual_elements = list()
inside_article = False
inside_content = False
full_articles_list = list()
cur_article = dict()
cur_article_text = list()
for idx, (event, elem) in enumerate(etree.iterparse(xml_path, events=('start', 'end', 'start-ns', 'end-ns'))):
    if len(full_articles_list) > articles_amount_limit:
        break
    # in case the tag is an article (start/end) - special handling
    if elem.tag == 'article':
        if event == 'start':
            #print(f"Found a new article named {elem.attrib['name']}")
            cur_article_text = list()
            cur_article = dict()
            cur_article['name'] = elem.attrib['name']
            inside_article = True
            continue
        else:
            cur_article_full_text = ''.join(cur_article_text)
            cur_article['content'] = cur_article_full_text
            # adding the article only in case the content is long enough (not a dummy wiki page)
            if len(cur_article['content']) > 1000:
                full_articles_list.append(cur_article)
                # printing status in another 10% chunk of the process has been finished
                if len(full_articles_list) > 0 and len(full_articles_list) % (articles_amount_limit / 10) == 0:
                    print(f"{len(full_articles_list)} wiki pages have been crawled "
                          f"({len(full_articles_list) / articles_amount_limit * 100}%)")
            inside_article = False
            continue
    # in case it is the beginning or end of content - we need to flag it
    if elem.tag == 'content':
        # do something
        inside_content = True if event == 'start' else False
        continue
    elif elem.tag == 'category':
        # do something
        pass
    # case it is a paragraph (simple text) - will add it as is
    elif elem.tag == 'p' and event == 'start' and inside_content:
        if elem.text is not None and elem.text != '\n':
            cur_article_text.append(elem.text)
        if elem.tail is not None and elem.tail != '\n':
            cur_article_text.append(elem.tail)
        continue
    # case it is a link - special handling (due to the tail of the link)
    elif elem.tag == 'link' and event == 'start' and inside_content:
        if elem.text is not None and elem.text != '\n':
            cur_article_text.append(elem.text)
        if elem.tail is not None and elem.tail != '\n':
            cur_article_text.append(elem.tail)
        continue
    # in case it is a header or math or
    elif elem.tag == 'h' and event == 'start' and inside_content:
        # we do not take the header, but only the start of the next paragraph
        if elem.tail is not None and elem.tail != '\n':
            cur_article_text.append(elem.tail)
        continue
    # other tags - 'table', 'cell', 'math'
    elif inside_content and event == 'start':
        continue

print(f"{len(full_articles_list)} wiki pages have been loaded successfully. The average number of chars in each "
      f"wiki page is: {np.mean([len(fal['content']) for fal in full_articles_list])}")
pickle.dump(full_articles_list, open(os.path.join(saving_path, saving_f_name), "wb"))
print(f"pickle file has been saved in {saving_path} with the name {saving_f_name}, now run the 'data_wikipedia' code")
