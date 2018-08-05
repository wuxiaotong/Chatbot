import json
from django.views.generic.base import TemplateView
from django.views.generic import View
from django.http import JsonResponse
from chatterbot import ChatBot
from chatterbot.ext.django_chatterbot import settings

class ChatterBotApiView(View):
    """
    Provide an API endpoint to interact with ChatterBot.
    """
    print('here1')
    chatterbot = ChatBot(**settings.CHATTERBOT)
    # chatterbot.train([
    # "Hi, can I help you?",
    # "Sure, I'd like to book a flight to Iceland.",
    # "Your flight has been booked."
    # ])
    def get_conversation(self, request):
        """
        Return the conversation for the session if one exists.
        Create a new conversation if one does not exist.
        """
        from chatterbot.ext.django_chatterbot.models import Conversation, Response

        class Obj(object):
            def __init__(self):
                self.id = None
                self.statements = []

        conversation = Obj()

        conversation.id = request.session.get('conversation_id', 0)
        existing_conversation = False
        try:
            Conversation.objects.get(id=conversation.id)
            existing_conversation = True

        except Conversation.DoesNotExist:
            conversation_id = self.chatterbot.storage.create_conversation()
            request.session['conversation_id'] = conversation_id
            conversation.id = conversation_id

        if existing_conversation:
            responses = Response.objects.filter(
                conversations__id=conversation.id
            )

            for response in responses:
                conversation.statements.append(response.statement.serialize())
                conversation.statements.append(response.response.serialize())

        return conversation

    def post(self, request, *args, **kwargs):
        """
        Return a response to the statement in the posted data.

        * The JSON data should contain a 'text' attribute.
        """
        print('post')
        input_data = json.loads(request.read().decode('utf-8'))
        print(input_data)
        if 'text' not in input_data:
            return JsonResponse({
                'text': [
                    'The attribute "text" is required.'
                ]
            }, status=400)

        conversation = self.get_conversation(request)
        print(input_data)
        response = self.chatterbot.get_response(input_data)
        print('response')
        response_data = response.serialize()

        return JsonResponse(response_data, status=200)

    def get(self, request, *args, **kwargs):
        """
        Return data corresponding to the current conversation.
        """
        print('here')
        conversation = self.get_conversation(request)

        return JsonResponse({
            'name': self.chatterbot.name,
            'conversation': conversation.statements
        })