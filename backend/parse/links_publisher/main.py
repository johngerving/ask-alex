from pdf_link_publisher import PDFLinkPublisher

def main():
    publisher = PDFLinkPublisher(
        publisher_queues=["links_queue"]
    )
    publisher.run()

if __name__ == "__main__":
    main()