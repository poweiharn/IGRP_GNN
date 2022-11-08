import json

class Args_JSON():
  def __init__(self, args):
    self.args = args

  def read(self, parser, file_name='temp_config.json'):
    with open(file_name, 'r') as f:
      parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    return args

  def export(self, args, file_name='temp_config.json'):
    tmp_args = vars(args).copy()
    # Directly from dictionary
    with open(file_name, 'w') as outfile:
      json.dump(tmp_args, outfile, indent=4)

  