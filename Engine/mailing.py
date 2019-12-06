from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

MY_ADDRESS = 'sandun.jayawardhana@yahoo.com'
PASSWORD = '********'
SMTP_SERVER = 'smtp.mail.yahoo.com'
PORT = 25
TO_ADDRESS = 'sandunmenaka@gmail.com'


def mail(address, message):
    s = smtplib.SMTP(host=SMTP_SERVER, port=PORT)
    s.connect(SMTP_SERVER, 25)
    s.starttls()
    s.ehlo()
    s.login(MY_ADDRESS, PASSWORD)
    msg = MIMEMultipart()
    msg['From'] = MY_ADDRESS
    msg['To'] = address
    msg['Subject'] = "This is TEST"
    msg.attach(MIMEText(message, 'plain'))
    s.send_message(msg)
    return 'Email sent'
