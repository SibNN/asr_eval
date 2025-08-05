from contextlib import contextmanager
from typing import Any, cast, override
import warnings

from langchain.agents import create_openai_tools_agent, AgentExecutor # type: ignore
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import SecretStr
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from .interfaces import TranscriptionCorrector


# def setup_ssl():
#     import ssl
#     # This will disable certificate verification globally
#     try:
#         ssl._create_default_https_context = (  # pyright:ignore[reportPrivateUsage]
#             ssl._create_unverified_context  # pyright:ignore[reportPrivateUsage]
#         )
#     except AttributeError:
#         pass


AGENT_SYSTEM_PROMPT_RU = '''\
Ты — профессиональный редактор и эксперт по исправлению транскрибаций аудио на русском языке. \
В твои задачи входит: 
1. Анализ текста транскрибации, выявление ошибок в терминах, именах и специфических фразах.
2. Проверка правильного написания терминов с помощью интернета.
3. Возврат полностью исправленного текста транскрибации.
В транскрибации могут быть ошибки, особенно в сложных терминах и именах. Найди ошибки и исправь \
их. Если термин вызывает сомнения — проверь его с помощью поиска. Если ты не находишь ошибок - \
ничего не меняй. Не добавляй никакие знаки препинания. Если нашёл ошибки, верни исправленную \
транскрибацию. Твоя задача - возвращать ТОЛЬКО транскрибацию и ничего больше. Не комментируй.\
'''

AGEMT_PROMPT_RU = '''\
Транскрибация аудио:

{transcription}\
'''


class CorrectorLangchain(TranscriptionCorrector):
    '''
    An agent that corrects a transcription, may use DuckDuckGo search.
    
    Author: Timur Rafikov
    Updated by: Oleg Sedukhin
    '''
    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.vsegpt.ru/v1',
        model_name: str = 'openai/gpt-4o',
        temperature: float = 0.3,
        verbose: bool = False,
        domain_specific_texts: str | None = None,
        use_web_search: bool = False,  # disabled by default, seems to work incorrectly (run with verbose=True)
    ):
        # setup_ssl()
        self.domain_specific_texts = domain_specific_texts
        self.llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=model_name,
            temperature=temperature,
        )
        self.tools: list[BaseTool] = []
        if use_web_search:
            self.search_tool = DuckDuckGoSearchRun(
                name='web_search',
                description='Поиск информации в интернете для уточнения терминов',
            )
            self.tools.append(self.search_tool)
        self.prompt = ChatPromptTemplate.from_messages([ # type: ignore
            ('system', AGENT_SYSTEM_PROMPT_RU),
            ('placeholder', '{chat_history}'),
            ('human', '{input}'),
            ('placeholder', '{agent_scratchpad}')
        ])
        self.agent = cast(
            Runnable[dict[str, Any], dict[str, Any]],
            create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt,
            )
        )
        print(type(self.agent))
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            max_iterations=10,
        )
    
    @override
    def correct(self, transcription: str) -> str:
        try:
            with ignore_warnings_specially_for_duckduckgo():
                result = self.agent_executor.invoke({
                    'input': AGEMT_PROMPT_RU.format(transcription=transcription)
                })
        except DuckDuckGoSearchException as e:
            print(repr(e))
            self.tools.remove(self.search_tool)
            result = self.agent_executor.invoke({
                'input': AGEMT_PROMPT_RU.format(transcription=transcription)
            })
            self.tools.append(self.search_tool)
            
        return result['output']
    

@contextmanager
def ignore_warnings_specially_for_duckduckgo():
    # duckduckgo_search sets warnings.simplefilter('ignore') and then prints a warning
    # "This package (`duckduckgo_search`) has been renamed to `ddgs`! Use `pip install ddgs` instead."
    warnings.filterwarnings("ignore", message=r'.*has been renamed to `ddgs`.*')
    simplefilter = warnings.simplefilter
    warnings.simplefilter = lambda *_args, **_kwargs: None # type: ignore
    yield None
    warnings.simplefilter = simplefilter