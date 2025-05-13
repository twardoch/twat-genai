# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # pkgs.go
    # pkgs.python311
    # pkgs.python311Packages.pip
    # pkgs.nodejs_20
    # pkgs.nodePackages.nodemon
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "beardedbear.beardedtheme"
      "bierner.markdown-image-size"
      "bmewburn.vscode-intelephense-client"
      "bradgashler.htmltagwrap"
      "charliermarsh.ruff"
      "christian-kohler.npm-intellisense"
      "codezombiech.gitignore"
      "continue.continue"
      "darkriszty.markdown-table-prettify"
      "davidanson.vscode-markdownlint"
      "dotjoshjohnson.xml"
      "ecmel.vscode-html-css"
      "esbenp.prettier-vscode"
      "fcrespo82.markdown-table-formatter"
      "fill-labs.dependi"
      "formulahendry.auto-close-tag"
      "foxundermoon.shell-format"
      "giga-ai.giga-ai"
      "github.vscode-github-actions"
      "github.vscode-pull-request-github"
      "hilleer.yaml-plus-json"
      "ionutvmi.path-autocomplete"
      "jebbs.markdown-extended"
      "jock.svg"
      "josee9988.minifyall"
      "kilocode.kilo-code"
      "lkrms.inifmt"
      "mblode.pretty-formatter"
      "mechatroner.rainbow-csv"
      "mikestead.dotenv"
      "mikoz.autoflake-extension"
      "mikoz.isort-formatter"
      "mrmlnc.vscode-scss"
      "ms-python.black-formatter"
      "ms-python.debugpy"
      "ms-python.flake8"
      "ms-python.isort"
      "ms-python.mypy-type-checker"
      "ms-python.pylint"
      "ms-toolsai.jupyter"
      "ms-vscode.cmake-tools"
      "ms-vscode.cpptools-themes"
      "ms-vscode.live-server"
      "ms-vscode.makefile-tools"
      "ms-vscode.powershell"
      "ms-windows-ai-studio.windows-ai-studio"
      "mtxr.sqltools"
      "oderwat.indent-rainbow"
      "pascalreitermann93.vscode-yaml-sort"
      "prateekmahendrakar.prettyxml"
      "qcz.text-power-tools"
      "redhat.vscode-yaml"
      "robole.marky-dynamic"
      "robole.marky-edit"
      "robole.marky-markdown"
      "robole.marky-stats"
      "rooveterinaryinc.roo-cline"
      "saketsarin.composer-web"
      "streetsidesoftware.code-spell-checker"
      "streetsidesoftware.code-spell-checker-polish"
      "stylelint.vscode-stylelint"
      "sylar.vscode-plugin-installer"
      "takumii.markdowntable"
      "timonwong.shellcheck"
      "twxs.cmake"
      "tyriar.sort-lines"
      "unifiedjs.vscode-remark"
      "visbydev.folder-path-color"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # web = {
        #   # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
        #   # and show it in IDX's web preview panel
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     # Environment variables to set for your server
        #     PORT = "$PORT";
        #   };
        # };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Example: install JS dependencies from NPM
        # npm-install = "npm install";
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";
      };
    };
  };
}
