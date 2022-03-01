import telegram as tel


class TelegramBot:

    def __init__(self):
        self.core = None
        self.updater = None
        self.id = None
        with open('private.txt', 'r') as f:
            data = f.read()
            data = data.split('\n')
            for i in data:
                if 'telegramtoken' in i:
                    token = i[i.find(':') + 1:]
                elif 'telegramchatid' in i:
                    chatid = i[i.find(':') + 1:]
        if token is None or chatid is None:
            return False
        self._set_chatid(chatid)
        self._set_token(token)
        self.bot = tel.Bot(token=self.token)
        self.sendmsg('텔레그램 로그인 완료')

    def _set_chatid(self, chat_id):
        self.id = chat_id

    def _set_token(self, token):
        self.token = token

    def sendmsg(self, text):
        self.bot.sendMessage(chat_id=self.id, text=text)


if __name__ == '__main__':
    tgBot = TelegramBot()
    tgBot.sendmsg('테스트')