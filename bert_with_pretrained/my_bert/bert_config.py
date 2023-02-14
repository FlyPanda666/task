import json
import copy
import six


class BertConfig:
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 pad_token_id=0,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """
        :param vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param num_hidden_layers: Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        :param pad_token_id:
        :param hidden_act: The non-linear activation function (function or string) in the encoder and pooler.
        :param hidden_dropout_prob: The dropout probability for all fully connected layers.
        :param attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        :param max_position_embeddings: The maximum sequence length that this model might ever be used with.
        :param type_vocab_size: The vocabulary size of the `token_type_ids` passed into `BertModel`.
        :param initializer_range: The st_dev of the truncated_normal_initializer for initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
