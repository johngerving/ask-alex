from pdf_parser import PDFParser

def main():
    parser = PDFParser(
        consumer_queue="links_queue",
        publisher_queues=["indexing_queue"],
        prefetch_count=1
    )
    parser.run()

if __name__ == "__main__":
    main()
